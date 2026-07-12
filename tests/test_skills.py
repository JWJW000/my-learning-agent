"""Tests for SkillsManager."""

import tempfile
from pathlib import Path

import pytest

from tools.skills_tool import SkillsManager


@pytest.fixture
def skills_manager(tmp_path):
    return SkillsManager(skills_dir=str(tmp_path / "skills"))


class TestSkillsManager:
    def test_create_and_list(self, skills_manager):
        result = skills_manager.create_skill(
            name="test-skill",
            description="A test skill",
            content="## Steps\n1. Do something\n2. Check result",
            category="testing",
        )
        assert "created" in result.lower()

        skills = skills_manager.list_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "test-skill"
        assert skills[0]["state"] == "active"

    def test_view_skill(self, skills_manager):
        skills_manager.create_skill(
            name="view-me", description="Viewable", content="## Content\nHello"
        )
        content = skills_manager.view_skill("view-me")
        assert content is not None
        assert "Hello" in content

    def test_view_bumps_usage(self, skills_manager):
        skills_manager.create_skill(
            name="bump-test", description="Test bump", content="Content"
        )
        skills_manager.view_skill("bump-test")
        skills_manager.view_skill("bump-test")

        skills = skills_manager.list_skills()
        assert skills[0]["use_count"] == 2

    def test_delete_archives(self, skills_manager):
        skills_manager.create_skill(
            name="delete-me", description="To delete", content="Gone"
        )
        result = skills_manager.delete_skill("delete-me")
        assert "archived" in result.lower()

        # Should not appear in active list
        active = skills_manager.list_skills(state="active")
        assert len(active) == 0

        # But should appear in archived
        archived = skills_manager.list_skills(state="archived")
        assert len(archived) == 1

    def test_name_validation(self, skills_manager):
        result = skills_manager.create_skill(
            name="a" * 100, description="Too long", content="X"
        )
        assert "error" in result.lower()

    def test_progressive_disclosure(self, skills_manager):
        skills_manager.create_skill(
            name="prog-test", description="Test", content="Long content here..."
        )
        summary = skills_manager.get_skills_summary()
        assert "prog-test" in summary
        assert "Long content" not in summary  # Only metadata, not full content

    def test_tool_schemas(self):
        from tools.registry import registry
        all_names = registry.get_all_tool_names()
        assert "skills_list" in all_names
        assert "skill_view" in all_names
        assert "skill_create" in all_names
        assert "skill_delete" in all_names

