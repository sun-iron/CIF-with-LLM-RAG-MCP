# client.py
# uv run client.py

import asyncio
import json
from fastmcp import Client

client = Client("http://localhost:8300")

def print_r(obj):
    print(json.dumps([t.model_dump() for t in obj], indent=4, ensure_ascii=False))

async def main():
    async with client:
        print(f"Client connected: {client.is_connected()}")
        
        print("Tools :")
        tools = await client.list_tools()
        print_r(tools)

        print("Prompts :")
        prompts = await client.list_prompts()
        print_r(prompts)

        print("Resources :")
        resources = await client.list_resources()
        print_r(resources)

        if any(tool.name == "execute_sql" for tool in tools):
            result = await client.call_tool("execute_sql", {"sql": "SELECT * FROM tr_result LIMIT 3;"})

            print(" ")
            print(f"execute_sql result: {result}")
            print(" ")

        if any(prompt.name == "get_prompt" for prompt in prompts):
            result = await client.get_prompt("get_prompt", {"task": "nl2sql"})

            print(" ")
            print(f"get_prompt result: {result}")
            print(" ")


    print(f"Client connected: {client.is_connected()}")

if __name__ == "__main__":
    asyncio.run(main())
