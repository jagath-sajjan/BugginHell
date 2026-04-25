from bughunt_env import BugHuntEnv


def main():
    env = BugHuntEnv(seed=7)
    obs, info = env.reset(seed=7)

    print("=== BugginHell Demo Episode ===")
    print("Case:", info["case_name"])
    print("Files:", obs.file_tree)
    print("Failing test:", obs.failing_test)
    print("Error:")
    print(obs.stderr)
    print()

    plan = [
        (2, {"name": obs.failing_test}),
        (3, {"fn": "calculate_total"}),
        (0, {"path": "cart.py"}),
        (4, {"file": "cart.py", "line": 3}),
    ]

    for action in plan:
        obs, reward, terminated, truncated, info = env.step(action)
        print("Action:", action)
        print("Reward:", reward)
        print("Output:")
        print(obs.last_tool_output)
        print("-" * 60)

        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
