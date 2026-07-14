"""Tests for YAML configuration loading."""

from agent.config import load_config


def test_load_config_reads_utf8_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """\
# 包含中文的 UTF-8 配置
model:
  primary: "测试模型"
""",
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.model.primary == "测试模型"
