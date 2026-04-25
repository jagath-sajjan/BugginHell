from bughunt_env import BugHuntEnv


def test_env_reset_works():
    env = BugHuntEnv(seed=1)
    obs, info = env.reset(seed=1)

    assert obs.file_tree
    assert obs.failing_test
    assert obs.stderr
    assert info["steps_used"] == 0


def test_correct_commit_gets_positive_reward():
    env = BugHuntEnv(seed=1)
    obs, info = env.reset(seed=1)

    target_file = env.case.bug_file
    target_line = env.case.bug_line

    obs, reward, terminated, truncated, info = env.step(
        (4, {"file": target_file, "line": target_line})
    )

    assert terminated is True
    assert truncated is False
    assert reward > 0


def test_wrong_commit_gets_negative_reward():
    env = BugHuntEnv(seed=1)
    obs, info = env.reset(seed=1)

    obs, reward, terminated, truncated, info = env.step(
        (4, {"file": "wrong.py", "line": 999})
    )

    assert terminated is True
    assert reward < 0


def test_budget_exhaustion_truncates():
    env = BugHuntEnv(max_steps=2, seed=1)
    obs, info = env.reset(seed=1)

    env.step((0, {"path": "README.md"}))
    obs, reward, terminated, truncated, info = env.step((0, {"path": "README.md"}))

    assert truncated is True
    assert reward < 0
