import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.infrastructure.llm.gemini_llm import GeminiLLM
from src.infrastructure.tools.echo_tool import EchoTool
from src.infrastructure.tools.search_tool import SimpleSearchTool
from src.infrastructure.tools.time_tool import TimeTool
from src.infrastructure.tool_registry import ToolRegistry
from src.infrastructure.logger import logger
from src.application.mcp_service import MCPService

# Load environment variables from .env file
load_dotenv()

logger.info("Starting MCP Service with Gemini...")

# Dependency Injection Setup
# 1. Initialize Infrastructure components
llm_provider = GeminiLLM(model_name="gemini-2.5-flash")

tool_registry = ToolRegistry()
tool_registry.register_tool(EchoTool())
tool_registry.register_tool(SimpleSearchTool())
tool_registry.register_tool(TimeTool())

# 2. Initialize Application service
mcp_service = MCPService(llm=llm_provider, tool_registry=tool_registry)

# FastAPI App
app = FastAPI(title="LangChain MCP Service MVP")

class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    result: str

@app.get("/")
def read_root():
    return {"message": "LangChain MCP Service is running"}

@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    try:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Google API Key not set")
            
        result = mcp_service.process_query(request.prompt)
        return QueryResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # For local testing without FastAPI server
    print("--- MCP Service CLI Mode ---")
    print("Registered Tools:", tool_registry.get_tool_names())
    
    # Simple CLI loop for quick verification
    # Note: Requires OPENAI_API_KEY
    # test_prompt = "What time is it now?"
    # print(f"Test Prompt: {test_prompt}")
    # print(f"Result: {mcp_service.process_query(test_prompt)}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
