from bughunt_env import ExternalBugHuntEnv
from server.source_loader import load_source


def test_load_source_from_pasted_files():
    spec = load_source(
        pasted_text="""=== app.py ===
def greet():
    return "hi"
"""
    )
    assert spec.source_type == "pasted_files"
    assert "app.py" in spec.files


def test_external_env_basic_episode():
    env = ExternalBugHuntEnv(
        files={
            "app.py": "def greet(name):\n    return f'hi {name}'\n",
            "tests/test_app.py": "from app import greet\n\ndef test_greet():\n    assert greet('x') == 'hi x'\n",
        },
        label="demo",
        source_type="pasted_files",
    )
    obs, info = env.reset(seed=1)
    assert obs.file_tree
    assert "EXTRACTED WORKSPACE MODE" in obs.stderr

    obs, reward, terminated, truncated, info = env.step((2, {"name": "greet"}))
    assert reward != 0
    assert terminated is False

    obs, reward, terminated, truncated, info = env.step((4, {"file": "app.py", "line": 1}))
    assert terminated is True
    assert "reward_breakdown" in info
