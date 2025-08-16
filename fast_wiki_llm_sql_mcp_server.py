from fastapi import FastAPI
from pydantic import BaseModel
import pymysql.cursors
import netifaces
import uvicorn

# 1. FastAPI 앱 초기화
app = FastAPI()

# 2. DB 게이트웨이 함수
def get_default_gateway():
    gateways = netifaces.gateways()
    default_gateway = gateways.get('default', {}).get(netifaces.AF_INET)
    return default_gateway[0] if default_gateway else "127.0.0.1"  # fallback

# 3. SQL 요청 모델 정의
class SQLRequest(BaseModel):
    sql: str

# 4. REST API 라우트로 SQL 실행 기능 등록
@app.post("/mcp/tools/execute_sql")
def execute_sql(req: SQLRequest):
    try:
        conn = pymysql.connect(
            host=get_default_gateway(),
            user="wpuser",
            password="1234",
            database="wordpress",
            port=3306,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True
        )
        cursor = conn.cursor()

        if not req.sql.strip().lower().startswith("select"):
            return {"error": "허용되지 않은 SQL입니다. SELECT 문만 실행 가능합니다."}

        cursor.execute(req.sql)
        return {"result": cursor.fetchall()}

    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()

# 메인 진입점 명시적으로 작성
if __name__ == "__main__":
    print(" REST 기반 MCP 서버 직접 실행 중")
    uvicorn.run("fast_wiki_llm_sql_mcp_server:app", host="0.0.0.0", port=8200, reload=True)
