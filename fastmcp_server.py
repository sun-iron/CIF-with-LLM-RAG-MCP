from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel
import sqlite3
import uvicorn
import logging
from typing import List

mcp = FastMCP(name="Domain Analysis System")

DB_PATH = "anlys_domain.db"

logging.basicConfig(filename="dom_debug.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)

@mcp.prompt(title = "Get server side prompt")
def get_prompt(task: str):
    """ take에 맞는 프롬프트를 제공합니다. task는 rag와 nl2sql이 있습니다. """
    prompts = {
        "rag": """
You are an assistant that answers questions using the provided context.
Be precise and avoid speculation. Use only the facts in the context.
Return concise, complete, and accurate information.

Context:
{context}

Question:
{question}

Answer:
""",
        "nl2sql": """
You are a SQL generation assistant.
You are given a natural language question, and your job is to translate it into a valid SQL SELECT statement.
You are working with a SQLite database containing a table named `tr_result` with the following columns:

- tableName: name of the database table
- callPath: calling function file path
- sourcePath: source path containing the call
- functionName: callee function name
- sourceName: trace code or source line

Important:
- Always use SELECT queries only.
- Always wrap table names and column names in square brackets ([]).
- Always use LOWER(column) and LIKE '%keyword%' style filters when matching text.
- Never return UPDATE, INSERT, DELETE, or DROP statements.

Example input:
"Find all tables related to login"

Output:
SELECT DISTINCT [tableName] FROM tr_result WHERE LOWER([sourceName]) LIKE '%login%' OR LOWER([functionName]) LIKE '%login%' OR LOWER([callPath]) LIKE '%login%' OR LOWER([sourcePath]) LIKE '%login%';

Question:
{question}

SQL:
"""
    }

    prompts = {
    "rag": """
당신은 주어진 문맥(Context)을 바탕으로 질문에 대답하는 어시스턴트입니다.
추측하지 말고, 문맥에 포함된 사실만을 근거로 정확하고 간결하게 대답하세요.

문맥:
{context}

질문:
{question}

답변:
""",
    
    "nl2sql": """
당신은 자연어를 SQL SELECT 문으로 변환하는 어시스턴트입니다.
다음은 `tr_result`라는 SQLite 테이블이며, 아래와 같은 컬럼을 포함하고 있습니다:

- tableName: 데이터베이스 테이블 이름
- callPath: 호출하는 함수의 전체 경로
- sourcePath: 호출이 포함된 소스 전체 경로
- functionName: 함수명
- sourceName: 소스파일명

중요 지침:
- 항상 SELECT 문만 사용하세요.
- 테이블명과 컬럼명은 반드시 대괄호([])로 감싸세요.
- 텍스트 필터링 시에는 반드시 LOWER(column)과 LIKE '%키워드%' 형태를 사용하세요.
- 절대로 UPDATE, INSERT, DELETE, DROP 문을 생성하지 마세요.
- 함수명을 검색할 때는 괄호 없이 `LIKE` 연산자를 사용하는 것이 좋습니다. (예: `LOWER(functionName) LIKE 'get_user_meta%'`)

예시 입력:
"로그인과 관련된 테이블을 찾아줘"
출력:
SELECT DISTINCT [tableName] FROM tr_result WHERE LOWER([sourceName]) LIKE '%login%' OR LOWER([functionName]) LIKE '%login%' OR LOWER([callPath]) LIKE '%login%' OR LOWER([sourcePath]) LIKE '%login%';

질문 의도에 따라 SELECT할 칼럼을 정확히 지정하세요:
- "어떤 테이블"을 물어보면 -> `SELECT DISTINCT [tableName]`
- "어떤 함수"를 물어보면 -> `SELECT DISTINCT [functionName]`
- "어떤 파일"을 물어보면 -> `SELECT DISTINCT [sourceName]`
- "파일 경로"를 물어보면 -> `SELECT DISTINCT [sourcePath]`

질문:
<question>

SQL:
"""
    }    
    if task not in prompts:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    return {"prompt": prompts[task]}


@mcp.tool(title = "Run SQL command")
def execute_sql(sql: str):
    """ SQL 명령어를 실행한 후 결과를 제공합니다. [e.g. SELECT * FROM tr_result]"""
    logger.debug(" ")
    logger.debug("   +++ (1) execute_sql: %s", sql)

    # sql = query.get("sql")

    if not sql or not sql.strip().lower().startswith("select"):
        raise HTTPException(status_code=400, detail="SELECT 명령만 사용할 수 있습니다.")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        result = [dict(zip(columns, row)) for row in rows]
        logger.debug("   +++ (2) execute_sql: %r", result)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8300,
        path="/",
        log_level="debug",
    )