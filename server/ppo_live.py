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

    lines = []
    rewards = []

    for row in logs:
        rewards.append(row["reward"])

        lines.append(f"EPISODE {row['episode']}")
        lines.append(f"Reward: {row['reward']:.2f}")
        lines.append(f"Loss: {row['loss']:.4f}")

        if row["actions"]:
            lines.append("Actions:")
            for i, action in enumerate(row["actions"]):
                action_id = action[0]
                params = action[1]
                lines.append(f"  {i + 1}. {ACTION_NAMES[action_id]} {params}")

        if row["probs"]:
            last_probs = row["probs"][-1]
            prob_text = ", ".join(
                f"{ACTION_NAMES[i]}={p:.2f}"
                for i, p in enumerate(last_probs)
            )
            lines.append(f"Final action probabilities: {prob_text}")

        lines.append("-" * 60)

    avg_last_5 = sum(rewards[-5:]) / max(1, len(rewards[-5:]))

    summary = [
        "LIVE PPO TRAINING COMPLETE",
        f"Episodes: {episodes}",
        f"Average reward last 5 episodes: {avg_last_5:.2f}",
        "",
    ]

    return "\n".join(summary + lines)
