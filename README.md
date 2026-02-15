# Personal Assistant MCP (MVP)

Clean/Layered Architecture 기반으로 "개인 비서형 AI"를 위한 Tool 중심 구조를 제공합니다.

## 핵심 목표

- `domain / application / infrastructure` 분리
- LLM 교체 가능(인터페이스 기반)
- LangChain 의존성은 `infrastructure`에만 존재
- Tool 중심(동적 등록, 독립 테스트 가능)
- 향후 DB/API/비동기 확장 고려

## 실행

- CLI 데모(기본: FakeLLM, 키가 있으면 Gemini 사용)
  - `python -m src.apps.cli_demo`
- HTTP 어댑터(FastAPI)
  - `python -m src.main`
  - `GET /tools` : 등록된 Tool 목록
  - `POST /tools/call` : Tool 직접 호출
  - `POST /query` : LLM 기반 Tool-calling 응답
  - `GET /mcp/tools`, `POST /mcp/tools/call`, `POST /mcp/query` : MCP 스타일 prefix

## 테스트

- `python -m unittest discover -s tests -p 'test_*.py'`

## 폴더 구조(요약)

- `src/domain` : 엔티티/인터페이스(LLM, Tool, Repository)
- `src/application` : 유즈케이스(MCPService), ToolRegistry, Tool 구현
- `src/infrastructure` : LangChain 어댑터(LLM), 저장소 구현(InMemory), 로깅 등
- `src/apps` : 실행 엔트리(예: CLI)
