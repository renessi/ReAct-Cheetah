import os
import pytest

@pytest.fixture(autouse=True)
def tmp_log(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_FILE_PATH", str(tmp_path / "agent.log"))