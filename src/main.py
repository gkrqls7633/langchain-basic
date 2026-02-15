import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.infrastructure.logger import logger
from src.application.mcp_service import MCPService
from src.application.tool_registry import ToolRegistry
from src.application.tools import CalculatorTool, EventDbTool, MemoTool, NationTool, SearchTool, TimeTool, TodoTool
from src.infrastructure.repositories.in_memory import InMemoryEventRepository, InMemoryMemoRepository, InMemoryTodoRepository
from src.infrastructure.mcp.server import MCPServerCore
from src.infrastructure.repositories.postgres import PostgresEventRepository
from src.infrastructure.llm.factory import build_llm_provider

# Load environment variables from .env file
load_dotenv()

logger.info("Starting MCP Service...")

# Dependency Injection Setup
# 1. Initialize Infrastructure components
llm_provider = build_llm_provider()

tool_registry = ToolRegistry()
todo_repo = InMemoryTodoRepository()
memo_repo = InMemoryMemoRepository()

# Event DB: 기본은 docker-compose Postgres로 시도 (DATABASE_URL 없으면 기본값으로 구성)
# 필요 시 EVENT_DB_MODE=memory 로 강제 in-memory 사용 가능
event_db_mode = os.getenv("EVENT_DB_MODE", "postgres").strip().lower()
if event_db_mode == "memory":
    event_repo = InMemoryEventRepository()
else:
    try:
        event_repo = PostgresEventRepository(database_url=os.getenv("DATABASE_URL"))
    except Exception as e:
        logger.warning("PostgresEventRepository init failed; falling back to InMemoryEventRepository: %s", e)
        event_repo = InMemoryEventRepository()

tool_registry.register(TimeTool())
tool_registry.register(TodoTool(todo_repo))
tool_registry.register(MemoTool(memo_repo))
tool_registry.register(SearchTool())
tool_registry.register(CalculatorTool())
tool_registry.register(NationTool())
tool_registry.register(EventDbTool(event_repo))


# 2. Initialize Application service
mcp_service = MCPService(llm=llm_provider, tool_registry=tool_registry)
mcp_core = MCPServerCore(tool_registry=tool_registry, assistant=mcp_service)

# FastAPI App
app = FastAPI(title="Personal Assistant MCP MVP (HTTP Adapter)")

class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    result: str

class ToolCallRequest(BaseModel):
    name: str
    args: dict = {}

@app.get("/")
def read_root():
    return {"message": "LangChain MCP Service is running"}

@app.get("/tools")
def list_tools():
    return mcp_core.list_tools()

@app.post("/tools/call")
def call_tool(request: ToolCallRequest):
    try:
        return {"result": mcp_core.call_tool(request.name, request.args)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    try:
        result = mcp_core.query(request.prompt)
        return QueryResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    # For local testing without FastAPI server
    print("--- MCP Service CLI Mode ---")
    print("Registered Tools:", tool_registry.names())
    
    # Simple CLI loop for quick verification
    # Note: Requires OPENAI_API_KEY
    # test_prompt = "What time is it now?"
    # print(f"Test Prompt: {test_prompt}")
    # print(f"Result: {mcp_service.process_query(test_prompt)}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
