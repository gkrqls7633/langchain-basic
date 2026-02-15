import os
import unittest
import uuid


def _sqlalchemy_available() -> bool:
    try:
        import sqlalchemy  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(_sqlalchemy_available(), "SQLAlchemy not available in this Python environment")
@unittest.skipUnless(os.getenv("RUN_PG_TESTS") == "1", "Set RUN_PG_TESTS=1 to run Postgres integration tests")
class PostgresIntegrationTests(unittest.TestCase):
    def test_postgres_insert_and_list(self):
        # Import here so the whole file can be imported even when deps are missing.
        from src.infrastructure.repositories.postgres import PostgresEventRepository

        # DATABASE_URL can be omitted; repository falls back to docker-compose defaults.
        repo = PostgresEventRepository(database_url=os.getenv("DATABASE_URL"))

        unique_kind = f"integration-{uuid.uuid4()}"
        inserted = repo.insert(kind=unique_kind, payload={"hello": "world"})

        events = repo.list(limit=50)
        found = next((e for e in events if e.id == inserted.id), None)
        self.assertIsNotNone(found)
        self.assertEqual(found.kind, unique_kind)
        self.assertEqual(found.payload.get("hello"), "world")


if __name__ == "__main__":
    unittest.main()

