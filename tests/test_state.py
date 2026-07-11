"""Tests for SessionStore (SQLite + FTS5)."""

import pytest

from agent.state import SessionStore


@pytest.fixture
def store(tmp_path):
    s = SessionStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


class TestSessionStore:
    def test_create_session(self, store):
        store.create_session("sess-1", model="gpt-4o")
        # Should not raise on duplicate
        store.create_session("sess-1", model="gpt-4o")

    def test_save_and_get_messages(self, store):
        store.create_session("sess-1")
        store.save_message("sess-1", "user", "Hello")
        store.save_message("sess-1", "assistant", "Hi there!")

        msgs = store.get_messages("sess-1")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["content"] == "Hi there!"

    def test_fts5_search(self, store):
        store.create_session("sess-1")
        store.save_message("sess-1", "user", "How do I debug Python tests?")
        store.save_message("sess-1", "assistant", "Use pytest with -x flag")

        store.create_session("sess-2")
        store.save_message("sess-2", "user", "What is JavaScript?")

        # Search should find Python-related messages
        results = store.search("Python debug")
        assert len(results) > 0
        assert any("Python" in r.get("snippet", "") for r in results)

    def test_update_session_summary(self, store):
        store.create_session("sess-1")
        store.update_session_summary("sess-1", "Discussed Python debugging")

        row = store.db.execute(
            "SELECT summary FROM sessions WHERE session_id = ?", ("sess-1",)
        ).fetchone()
        assert row[0] == "Discussed Python debugging"

    def test_token_accumulation(self, store):
        store.create_session("sess-1")
        store.update_session_tokens("sess-1", 100, 50)
        store.update_session_tokens("sess-1", 200, 100)

        row = store.db.execute(
            "SELECT input_tokens, output_tokens FROM sessions WHERE session_id = ?",
            ("sess-1",),
        ).fetchone()
        assert row[0] == 300
        assert row[1] == 150
