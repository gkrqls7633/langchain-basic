CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at DESC);

