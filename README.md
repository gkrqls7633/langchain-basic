# Personal Assistant MCP (MVP)

Tool 중심 개인 비서형 AI 서버(MVP)입니다. Clean/Layered Architecture로 `domain / application / infrastructure`를 분리했고, LLM은 `Gemini` 또는 `로컬 Ollama` 중 하나로 쉽게 교체할 수 있습니다.

## 아키텍처 한눈에 보기

- **Domain**: 순수 인터페이스/모델
  - `src/domain/tool.py` (Tool 인터페이스)
  - `src/domain/llm.py` (LLMProvider 인터페이스)
  - `src/domain/repositories.py` (Repository 인터페이스)
- **Application**: 유즈케이스 + Tool 구현
  - `src/application/mcp_service.py` (LLM ↔ Tools 오케스트레이션)
  - `src/application/tool_registry.py` (Tool 등록/조회)
  - `src/application/tools/` (Time/Todo/Memo/Search/Calculator/Nation/EventDb)
- **Infrastructure**: 외부 의존성(LangChain, DB, HTTP)
  - `src/infrastructure/llm/gemini_llm.py`, `src/infrastructure/llm/ollama_llm.py`
  - `src/infrastructure/llm/factory.py` (LLM 선택: Gemini vs Ollama)
  - `src/infrastructure/repositories/postgres.py` (Postgres)
  - `src/main.py` (FastAPI HTTP adapter)

요청 흐름:
- `POST /query` → `MCPService.process_query()` → `LLMProvider.generate(tools=...)`
- LangChain agent가 tool을 선택/호출 → tool 결과 기반으로 최종 응답 생성

## Tool 호출 이력(DB)

`event_db_tool`이 등록되어 있으면, `GeminiLLM`/`OllamaLLM`의 callback에서 tool 호출/최종 응답을 자동으로 `events` 테이블에 저장합니다.
- kind=`tool_call`: 어떤 tool을 어떤 인자로 호출했고 결과가 무엇인지
- kind=`final`: 최종 응답

## 빠른 시작

1) (선택) Docker 서비스 실행
- Postgres: `docker-compose up -d postgres`
- Ollama: `docker-compose up -d ollama`

2) (선택) Ollama 모델 pull
- 예: `docker-compose exec ollama ollama pull lfm2.5-thinking:1.2b`
- 기존 모델 회수 : `docker-compose exec ollama ollama rm qwen2.5:1.5b`

3) Python 실행
- `./venv/bin/python -m src.main` (venv 사용 권장)
- FastAPI docs: `http://localhost:8000/docs`

## 환경변수(중요)

- LLM 선택
  - `LLM_BACKEND=gemini|ollama`
  - Gemini: `GEMINI_API_KEY` (또는 `GOOGLE_API_KEY`), `GEMINI_MODEL`(optional)
  - Ollama: `OLLAMA_BASE_URL`(default `http://localhost:11434`), `OLLAMA_MODEL`(default `lfm2.5-thinking:1.2b`)
- DB(Event 저장)
  - `EVENT_DB_MODE=postgres|memory` (default `postgres`)
  - `DATABASE_URL` (없으면 docker-compose 기본값으로 조합)

## API

- `GET /tools` : 등록된 Tool 목록
- `POST /tools/call` : Tool 직접 호출
- `POST /query` : LLM 기반 tool-calling 응답

## DB(Postgres)

- init SQL: `db/init/001_create_events.sql` → `events` 테이블 생성
- 기존 `pgdata` 볼륨이 있으면 init SQL은 재실행되지 않습니다.
- 그래도 `event_db_tool`을 호출하면 코드가 `create_all()`로 테이블을 보장합니다.

## 테스트

- 유닛 테스트: `./venv/bin/python -m unittest discover -s tests -p 'test_*.py'`
- (선택) Postgres 통합 테스트:
  - `docker-compose up -d postgres`
  - `RUN_PG_TESTS=1 python -m unittest tests.test_postgres_integration -v`
