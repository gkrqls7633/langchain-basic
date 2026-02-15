# Personal Assistant MCP (MVP)

Clean/Layered Architecture 기반으로 "개인 비서형 AI"를 위한 Tool 중심 구조를 제공합니다.

## 핵심 목표

- `domain / application / infrastructure` 분리
- LLM 교체 가능(인터페이스 기반)
- LangChain 의존성은 `infrastructure`에만 존재
- Tool 중심(동적 등록, 독립 테스트 가능)
- 향후 DB/API/비동기 확장 고려

## 실행

- Postgres(Docker)
  - `docker-compose up -d`
  - 기본 테이블: `events` (docker init SQL: `db/init/001_create_events.sql`)
  - 연결 문자열 예시: `DATABASE_URL=postgresql+psycopg2://assistant:assistant@localhost:5432/assistant`
  - `DATABASE_URL`이 없으면 코드가 위 기본값으로 접속을 시도합니다(로컬 docker 기준).
  - 기존 `pgdata` 볼륨이 있으면 init SQL은 재실행되지 않습니다. 이때는 `event_db_tool`을 한 번 호출하면(create_all) 테이블이 생성됩니다.

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
- (선택) Postgres 통합 테스트(로컬 docker 필요)
  - `docker-compose up -d`
  - `RUN_PG_TESTS=1 python -m unittest tests.test_postgres_integration -v`

## 폴더 구조(요약)

- `src/domain` : 엔티티/인터페이스(LLM, Tool, Repository)
- `src/application` : 유즈케이스(MCPService), ToolRegistry, Tool 구현
- `src/infrastructure` : LangChain 어댑터(LLM), 저장소 구현(InMemory), 로깅 등
- `src/apps` : 실행 엔트리(예: CLI)

## 서버 구동
- ./venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8080
