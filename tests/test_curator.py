"""Tests for Curator skill lifecycle management."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from agent.curator import Curator
from tools.skills_tool import SkillsManager


@pytest.fixture
def setup(tmp_path):
    """Create skills manager and curator with temp dirs."""
    skills_dir = tmp_path / "skills"
    state_file = tmp_path / ".curator_state"

    sm = SkillsManager(skills_dir=str(skills_dir))
    curator = Curator(
        skills_dir=str(skills_dir),
        state_file=str(state_file),
        interval_hours=0,  # Always ready to run in tests
        stale_after_days=30,
        archive_after_days=90,
    )
    return sm, curator


def _set_last_used(skill_path: str, days_ago: int):
    """Helper: set a skill's last_used to N days ago."""
    path = Path(skill_path)
    content = path.read_text()
    parts = content.split("---", 2)
    fm = yaml.safe_load(parts[1])
    fm["metadata"]["last_used"] = (datetime.now() - timedelta(days=days_ago)).isoformat()
    path.write_text(
        f"---\n{yaml.dump(fm, allow_unicode=True, default_flow_style=False)}---{parts[2]}"
    )


class TestCurator:
    def test_should_run_first_time_defers(self, setup):
        _, curator = setup
        # First call seeds state and defers
        assert curator.should_run_now() is False

    def test_active_to_stale(self, setup):
        sm, curator = setup
        sm.create_skill(name="old-skill", description="Old", content="Content")

        skills = sm.list_skills()
        _set_last_used(skills[0]["path"], days_ago=35)

        result = curator.apply_automatic_transitions()
        assert "old-skill" in result["staled"]

    def test_stale_to_archived(self, setup):
        sm, curator = setup
        sm.create_skill(name="ancient-skill", description="Ancient", content="Content")

        skills = sm.list_skills()
        path = Path(skills[0]["path"])
        # Set to stale first
        fm = SkillsManager._parse_frontmatter(path)
        fm["metadata"]["state"] = "stale"
        SkillsManager._write_frontmatter(path, fm)

        _set_last_used(skills[0]["path"], days_ago=95)

        result = curator.apply_automatic_transitions()
        assert "ancient-skill" in result["archived"]

    def test_pinned_skill_untouched(self, setup):
        sm, curator = setup
        sm.create_skill(name="pinned-skill", description="Pinned", content="Content")

        skills = sm.list_skills()
        path = Path(skills[0]["path"])
        fm = SkillsManager._parse_frontmatter(path)
        fm["metadata"]["pinned"] = True
        SkillsManager._write_frontmatter(path, fm)
        _set_last_used(skills[0]["path"], days_ago=100)

        result = curator.apply_automatic_transitions()
        assert len(result["staled"]) == 0
        assert len(result["archived"]) == 0

    def test_restore_skill(self, setup):
        sm, curator = setup
        sm.create_skill(name="restore-me", description="Restore", content="Content")
        sm.delete_skill("restore-me")

        result = curator.restore_skill("restore-me")
        assert "restored" in result.lower()

        # Should be active again
        skills = sm.list_skills(state="active")
        assert any(s["name"] == "restore-me" for s in skills)

    def test_run_updates_state(self, setup):
        _, curator = setup
        curator._state["last_run_at"] = (datetime.now() - timedelta(hours=200)).isoformat()
        curator._save_state()

        summary = curator.run()
        assert curator._state["run_count"] == 1
