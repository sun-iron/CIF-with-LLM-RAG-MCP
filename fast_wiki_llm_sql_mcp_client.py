from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import logging
import re
import httpx

app = FastAPI()

logging.basicConfig(filename="mcp_debug.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 모델 초기화
llm = Llama(
    model_path="../models/ggml-model-q4_k_m.gguf",
    n_ctx=4096,
    n_threads=6
)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("./data/wiki_it.index")
docs = pickle.load(open("./data/wiki_it.pkl", "rb"))

class AskRequest(BaseModel):
    prompt: str
    max_tokens: int = 512

def get_intent_from_llm(user_query: str) -> str:
    intent_prompt = f"""
너는 회사 인사DB 시스템의 의도 분류기이다.

아래 규칙으로 사용자의 질문을 분류해라:
- SQL: 직원, 사원, 재직, 퇴사, 부서, 직급, 입사일, 퇴사일, 퇴사상태, 로그, 로그인, 로그아웃 등 회사 시스템 인사DB와 관련된 정보는 무조건 SQL로 분류한다.
- RAG: 회사 정책, 문화, 역사, IT기술 등 DB에 저장되지 않은 정보는 RAG로 분류한다.
- LLM: 단순 인사, 농담, 설명은 LLM으로 분류한다.

예시:
- "모든 직원 목록 보여줘" → SQL
- "디지인팀 직원목록" → SQL
- "홍길동의 입사일?" → SQL
- "SQL을 작성해줘" → SQL
- "우리 회사의 비전은?" → RAG
- "안녕?" → LLM

아래 질의를 SQL, RAG, LLM 중 하나의 단어로만 답해라.
답변은 반드시 다음 중 하나의 단어만 출력하세요: SQL, RAG, LLM.
다른 문장, 기호, 설명 없이 단어 하나만 출력하세요.

이제 다음 질문을 분류해 보세요:
Q: "{user_query}"
A:
"""
    response = llm(
        intent_prompt,
        max_tokens=10,
        temperature=0.0,
        stop=["\n"],
        top_p=1.0,
        top_k=0,
        repeat_penalty=1.0
    )
    raw_output = response["choices"][0]["text"]
    logger.debug("   +++ (1) raw intent output: %r", raw_output)

    # 방어적 처리
    if not raw_output.strip():
        logger.warning("   !!! intent 판단 실패 → fallback to LLM")
        return "LLM"

    # 정상 파싱
    result = re.sub(r"[^A-Z]", "", raw_output.upper())
    logger.debug("   +++ (2) cleaned intent: %s", result)

    return result if result in ["SQL", "RAG", "LLM"] else "LLM"

def retrieve_context_sliding(query, k=5):
    vec = embedder.encode([query])
    D, I = index.search(vec, k)
    seen = set()
    unique_chunks = []
    for i in I[0]:
        chunk = docs[i].strip()
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)
    return "\n".join(unique_chunks)

def remove_duplicate_sentences(text: str) -> str:
    seen = set()
    result = []
    sentences = re.split(r'(?:\n+)|(?<=\d[).])\s+|(?<=[.!?])\s+', text)
    for sentence in sentences:
        cleaned = sentence.strip()
        if cleaned and cleaned not in seen:
            result.append(cleaned)
            seen.add(cleaned)
    return ' '.join(result)

def generate_sql_from_prompt(query: str) -> str:
    table_description = """
- mcp_users(employee_id, name, department, position, join_date, resign_date)
- mcp_access_logs(employee_id, login_time, logout_time)
"""
    employment_rule = """
- 재직: resign_date = '9999-12-31'
- 퇴직: resign_date != '9999-12-31'
- NULL 비교는 금지
"""
    prompt_to_sql = f"""
다음은 회사 인사 DB를 위한 자연어 질의입니다.

{table_description}

{employment_rule}

아래 질문을 SQL로 변환해주세요. 조건:
1. 질문에 맞는 필터 적용
2. 출력 필드 순서는 질문 순서
3. SELECT 문만 생성

질문: \"{query}\"
SQL:
"""
    sql_response = llm(
        prompt_to_sql,
        max_tokens=512,
        temperature=0.0,
        stop=["\n\n질문:"],
        top_p=1.0,
        top_k=0,
        repeat_penalty=1.0
    )
    return sql_response["choices"][0]["text"].strip()

async def execute_sql_via_mcp(sql: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8200/mcp/tools/execute_sql",
            json={"sql": sql}
        )
        
        if response.status_code != 200:
            return f"MCP 서버 오류: {response.text}"
        result = response.json()
        if "error" in result:
            return f"SQL 실행 오류: {result['error']}"
        if not result:
            return "결과가 없습니다."

        rows = result.get("result", [])
        if not rows:
            return "결과가 없습니다(2)."

        # 결과 렌더링
        keys = rows[0].keys()
        lines = [", ".join(keys)]
        for row in rows:
            lines.append(", ".join(str(row[k]) for k in keys))
        
        return "\n".join(lines)

@app.post("/ask")
async def ask(req: AskRequest):
    prompt = req.prompt.strip()
    logger.debug("[[[[ 질의문 : %s ]]]]", prompt)

    intent = get_intent_from_llm(prompt)
    logger.debug(">>>> 분류된 intent: %s", intent)

    if intent == "SQL":
        sql = generate_sql_from_prompt(prompt)
        result = await execute_sql_via_mcp(sql)
        return {"response": result, "source": "MCP_SQL"}

    elif intent == "RAG":
        context = retrieve_context_sliding(prompt)
        full_prompt = f"""
다음은 질문에 대한 기술 문서 정보입니다. 이를 바탕으로 답변을 작성하세요:

{context}

질문: {prompt}
답변:"""
    else:
        full_prompt = f"사용자 질문에 중복 없이 10문장 이하로 상세히 설명해 주세요.\n\n질문: {prompt}\n답변:"

    output = llm(
        full_prompt,
        max_tokens=req.max_tokens,
        temperature=0.0,
        stop=["\n질문:"],
        top_p=1.0,
        top_k=0,
        repeat_penalty=1.3
    )
    raw_answer = output["choices"][0]["text"]
    cleaned_answer = remove_duplicate_sentences(raw_answer)
    return {"response": cleaned_answer.strip(), "source": intent}


@app.post("/ask/llm")
async def ask(req: AskRequest):
    prompt = req.prompt.strip()
    logger.debug("[[[[ 질의문 : %s ]]]]", prompt)
    full_prompt = f"사용자 질문에 중복 없이 10문장 이하로 상세히 설명해 주세요.\n\n질문: {prompt}\n답변:"

    output = llm(
        full_prompt,
        max_tokens=req.max_tokens,
        temperature=0.0,
        stop=["\n질문:"],
        top_p=1.0,
        top_k=0,
        repeat_penalty=1.3
    )
    raw_answer = output["choices"][0]["text"]
    cleaned_answer = remove_duplicate_sentences(raw_answer)
    return {"response": cleaned_answer.strip(), "source": "LLM"}


@app.post("/ask/rag")
async def ask(req: AskRequest):
    prompt = req.prompt.strip()
    logger.debug("[[[[ 질의문 : %s ]]]]", prompt)
    context = retrieve_context_sliding(prompt)
    full_prompt = f"""
다음은 질문에 대한 기술 문서 정보입니다. 이를 바탕으로 답변을 작성하세요:

{context}

질문: {prompt}
답변:"""

    output = llm(
        full_prompt,
        max_tokens=req.max_tokens,
        temperature=0.0,
        stop=["\n질문:"],
        top_p=1.0,
        top_k=0,
        repeat_penalty=1.3
    )
    raw_answer = output["choices"][0]["text"]
    cleaned_answer = remove_duplicate_sentences(raw_answer)
    return {"response": cleaned_answer.strip(), "source": "RAG"}
