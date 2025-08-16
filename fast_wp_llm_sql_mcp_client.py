from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import logging
import re
import httpx
from typing import List

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

wp_index = faiss.read_index("./data/rag_wp.index")
wp_docs = pickle.load(open("./data/rag_wp.pkl", "rb"))


class AskRequest(BaseModel):
    prompt: str
    max_tokens: int = 512

def get_intent_from_llm(user_query: str) -> str:
    intent_prompt = f"""
너는 회사 시스템의 의도 분류기이다.

아래 규칙으로 사용자의 질문을 분류해라:
- SQL: 직원, 사원, 재직, 퇴사, 부서, 직급, 입사일, 퇴사일, 퇴사상태, 로그, 로그인, 로그아웃 등 인사DB에 저장된 정보는 무조건 SQL로 분류한다.
- RAG: 회사 정책, 문화, 역사, IT기술 등 DB에 저장되지 않은 일반 정보는 RAG로 분류한다.
- SYS: "분석시스템"이라는 단어가 포함되어 있다면 **무조건 SYS**으로 분류한다.  
        특히, 도메인 구조, 테이블 역할, 함수와 기능의 관계 분석 등은 SYS으로 분류한다.
- LLM: 단순 인사, 농담, 설명 등은 LLM으로 분류한다.

예시:
- "모든 직원 목록 보여줘" → SQL
- "디지인팀 직원목록" → SQL
- "홍길동의 입사일?" → SQL
- "SQL을 작성해줘" → SQL
- "우리 회사의 비전은?" → RAG
- "분석시스템" → SYS
- "분석시스템에서 기능과 관련된 함수명을 보여줘?" → SYS
- "분석시스템에서 기능과 관련된 테이블은 어떤거야?" → SYS
- "분석시스템에서 로그인 관련 테이블은?" → SYS
- "분석시스템에서 사용자 로그인과 관련된 테이블 목록은?" → SYS
- "안녕?" → LLM

답변은 반드시 다음 중 하나의 단어만 출력하세요: SQL, RAG, SYS, LLM.
다른 문장, 기호, 설명 없이 단어 하나만 출력하세요.

이제 다음 질문을 분류해 보세요:
- "{user_query}"
"""
    response = llm(
        intent_prompt,
        max_tokens=8,
        temperature=0.0,
        stop=["\n"],
        top_p=1.0,
        top_k=0,
        repeat_penalty=1.0
    )
    raw_output = response["choices"][0]["text"]
    logger.debug("   +++ (1) raw intent output: %r", raw_output)

    # 클린업
    cleaned = raw_output.upper().strip()
    cleaned = re.sub(r"[^A-Z]", "", cleaned)

    # 방어적 처리
    if cleaned not in {"SQL", "RAG", "SYS", "LLM"}:
        logger.warning("   !!! intent 판단 실패 → fallback to LLM")
        logger.debug("   >>> intent_prompt was: %s", intent_prompt)
        return "LLM"

    # 정상 파싱
    logger.debug("   +++ (2) cleaned intent: %s", cleaned)

    return cleaned

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
    elif intent == "DOM":
        context = await get_analysis_domain(prompt)
        full_prompt = f"""
다음은 질문에 대한 분석결과 정보입니다. 이를 바탕으로 답변을 작성하세요:

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


@app.post("/ask/dom")
async def ask(req: AskRequest):
    prompt = req.prompt.strip()
    logger.debug("[[[[ 질의문 : %s ]]]]", prompt)

    # 1. 미리 정의된 search_xxx 함수 호출 여부 판단
    predef_result = await call_predefined_search_function(prompt)
    if predef_result:
        return {"response": predef_result, "source": "MCP_SEARCH"}

    # 2. 아니면 → RAG 문서 기반 NL2SQL 실행
    result = await generate_sql_for_dom(prompt)
    return {
        "response": result,
        "source": "DOM"
    }


# 프롬프트를 MCP 서버로부터 가져오기
async def get_prompt_from_mcp(task: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"http://localhost:8300/mcp/prompts/{task}")
        return r.json().get("prompt", "")


# 검색 키워드 추출
def extract_search_keyword(prompt: str) -> str:
    import re
    words = re.findall(r"[가-힣a-zA-Z0-9_]+", prompt)

    # 영어 기준으로 의미 없는 단어 제거
    stopwords = {
        "the", "is", "are", "and", "or", "of", "to", "in", "related", "with",
        "list", "information", "about", "what", "which", "show", "get"
    }
    candidates = [w for w in words if w.lower() not in stopwords]

    for w in reversed(candidates):
        return w  # 마지막 의미 있는 단어 반환

    return "user"


# search_xxx 함수 호출
async def call_search_api(endpoint: str, keyword: str) -> List[str]:
    url = f"http://localhost:8300/mcp/dom/{endpoint}?keyword={keyword}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        return (
            resp.json().get("tables", []) or
            resp.json().get("functions", []) or
            resp.json().get("files", [])
        )


# 사전 정의된 검색 함수 실행 여부 확인
async def call_predefined_search_function(prompt: str) -> str:
    prompt_lower = prompt.lower()
    keyword = extract_search_keyword(prompt)

    if "table" in prompt_lower and "related" in prompt_lower:
        return await call_search_api("search_tables", keyword)
    elif "function" in prompt_lower and "related" in prompt_lower:
        return await call_search_api("search_functions", keyword)
    elif "file" in prompt_lower and "related" in prompt_lower:
        return await call_search_api("search_files", keyword)

    return None


# RAG 문서 기반 SQL 생성
async def generate_sql_for_dom(prompt: str) -> str:
    prompt_template = await get_prompt_from_mcp("nl2sql")

    logger.debug(">> [1] prompt: %r", prompt)
    logger.debug(">> [2] prompt_template: %r", prompt_template)

    filled_prompt = prompt_template.replace("<question>", prompt)

    logger.debug(">> [3] filled_prompt: %r", filled_prompt)

    sql_response = llm(
        filled_prompt,
        max_tokens=512,
        temperature=0.0,
        stop=["\n\n질문:", "SQL:"],
        top_p=1.0,
        top_k=0,
        repeat_penalty=1.0
    )
    sql = sql_response["choices"][0]["text"].strip()

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8300/mcp/tools/execute_sql",
            json={"sql": sql}
        )
        if response.status_code != 200:
            return f"MCP 서버 오류: {response.text}"
        
        result = response.json()

        if "error" in result:
            return f"SQL 실행 오류: {result['error']}"
        if not result:
            return "결과가 없습니다."
        
        logger.debug(">> [4] result: %r", result)

        if isinstance(result, list):
            rows = result
        elif isinstance(result, dict):
            rows = result.get("result", [])
        else:
            return "결과가 없습니다(2)."

        keys = rows[0].keys()
        lines = [", ".join(keys)]
        for row in rows:
            lines.append(", ".join(str(row[k]) for k in keys))
        return "\n".join(lines)