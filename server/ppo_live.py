import pandas as pd
import matplotlib.pyplot as plt

from bughunt_env import BugHuntEnv, PPOTrainer


ACTION_NAMES = [
    "read_file",
    "run_test",
    "search_symbol",
    "trace_caller",
    "commit_location",
]


def run_live_ppo_training(episodes=25):
    trainer = PPOTrainer(BugHuntEnv)
    logs = trainer.train(episodes=int(episodes))

    rows = []
    text_lines = []

    for row in logs:
        probs = row["probs"][-1] if row["probs"] else [0, 0, 0, 0, 0]

        rows.append({
            "episode": row["episode"],
            "reward": row["reward"],
            "loss": row["loss"],
            "read_file": probs[0],
            "run_test": probs[1],
            "search_symbol": probs[2],
            "trace_caller": probs[3],
            "commit_location": probs[4],
        })

        text_lines.append(f"EPISODE {row['episode']} | reward={row['reward']:.2f} | loss={row['loss']:.4f}")

        if row["actions"]:
            readable_actions = []
            for action in row["actions"]:
                action_id, params = action
                readable_actions.append(f"{ACTION_NAMES[action_id]}{params}")
            text_lines.append(" → ".join(readable_actions))

        text_lines.append("-" * 80)

    df = pd.DataFrame(rows)

    reward_fig = plt.figure()
    plt.plot(df["episode"], df["reward"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Live PPO Reward Curve")

    loss_fig = plt.figure()
    plt.plot(df["episode"], df["loss"])
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Live PPO Loss Curve")

    return "\n".join(text_lines), df, reward_fig, loss_fig
