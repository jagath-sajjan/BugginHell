from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bughunt_env import BugHuntEnv, PPOTrainer
from bughunt_env.ppo_agent import ACTION_NAMES
from server.neural_viz import render_neural_network_svg


def rolling_success(successes: Sequence[bool], window: int = 25) -> List[float]:
    values: List[float] = []
    for idx in range(len(successes)):
        start = max(0, idx - window + 1)
        segment = successes[start : idx + 1]
        values.append(sum(bool(item) for item in segment) / max(1, len(segment)))
    return values


def action_distribution(logs: Sequence[dict]) -> pd.DataFrame:
    rows = []
    for row in logs:
        counts = {name: 0 for name in ACTION_NAMES}
        for action_id, _ in row["actions"]:
            counts[ACTION_NAMES[action_id]] += 1
        total = max(1, sum(counts.values()))
        rows.append({"episode": row["episode"], **{name: counts[name] / total for name in ACTION_NAMES}})
    return pd.DataFrame(rows)


def case_action_heatmap(logs: Sequence[dict]) -> pd.DataFrame:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {name: 0 for name in ACTION_NAMES})
    for row in logs:
        case_name = row.get("case_name") or "unknown_case"
        for action_id, _ in row["actions"]:
            counts[case_name][ACTION_NAMES[action_id]] += 1
    if not counts:
        return pd.DataFrame(0, index=["no_data"], columns=ACTION_NAMES)
    return pd.DataFrame.from_dict(counts, orient="index")[ACTION_NAMES].sort_index()


def build_dashboard_rows(logs: Sequence[dict]) -> pd.DataFrame:
    rows = []
    action_mix = action_distribution(logs)
    rolling = rolling_success([row["success"] for row in logs], window=25)
    for idx, row in enumerate(logs):
        base = {
            "episode": row["episode"],
            "reward": row["reward"],
            "loss": row["loss"],
            "success": row["success"],
            "rolling_success_25": rolling[idx],
            "entropy": row["entropy"],
            "case_name": row.get("case_name"),
        }
        if not action_mix.empty:
            for name in ACTION_NAMES:
                base[name] = float(action_mix.iloc[idx][name])
        rows.append(base)
    return pd.DataFrame(rows)


def plot_reward_curve(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(df["episode"], df["reward"], color="#e85d04", linewidth=2)
    ax.set_title("Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_loss_curve(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(df["episode"], df["loss"], color="#4361ee", linewidth=2)
    ax.set_title("Loss")
    ax.set_xlabel("Episode")
    ax.set_ylabel("PPO loss")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_success_curve(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(df["episode"], df["rolling_success_25"], color="#2a9d8f", linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_title("Rolling Success Rate (25)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_action_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = df["episode"].to_numpy()
    ys = [df[name].to_numpy() for name in ACTION_NAMES]
    ax.stackplot(x, ys, labels=ACTION_NAMES, alpha=0.9)
    ax.set_title("Action Distribution")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Share")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_entropy_curve(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(df["episode"], df["entropy"], color="#8d5cf6", linewidth=2)
    ax.set_title("Policy Entropy")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Entropy")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_case_action_heatmap(heatmap_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.4 * len(heatmap_df.index) + 1.5)))
    image = ax.imshow(heatmap_df.to_numpy(), cmap="magma", aspect="auto")
    ax.set_xticks(range(len(heatmap_df.columns)), labels=heatmap_df.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(heatmap_df.index)), labels=heatmap_df.index)
    ax.set_title("Actions per Bug Case")
    for i in range(len(heatmap_df.index)):
        for j in range(len(heatmap_df.columns)):
            ax.text(j, i, int(heatmap_df.iloc[i, j]), ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig


def summarize_logs(logs: Sequence[dict]) -> str:
    lines: List[str] = []
    for row in logs:
        action_chain = " -> ".join(ACTION_NAMES[action_id] for action_id, _ in row["actions"])
        lines.append(
            f"EP {row['episode']:03d} | case={row.get('case_name')} | reward={row['reward']:.2f} | "
            f"loss={row['loss']:.4f} | entropy={row['entropy']:.3f} | success={row['success']}"
        )
        lines.append(action_chain or "<no actions>")
        lines.append("-" * 92)
    return "\n".join(lines)


def render_dashboard_viz(row: dict) -> str:
    activations = row.get("activations") or []
    if not activations:
        return "<div style='padding:18px;border:1px solid rgba(255,255,255,0.1);border-radius:16px;background:#08111f;color:#dbe7f6;font:14px monospace;'>No activation snapshot available yet.</div>"
    snapshot = activations[-1]
    return render_neural_network_svg(
        features=snapshot["features"],
        h1=snapshot["h1"],
        h2=snapshot["h2"],
        logits=snapshot["logits"],
        probs=snapshot["probs"],
        c1=snapshot["c1"],
        c2=snapshot["c2"],
        c3=snapshot["c3"],
        value_estimate=snapshot["value"],
        step_index=max(0, len(activations) - 1),
        episode_index=row["episode"],
        feature_names=snapshot["feature_names"],
        action_id=snapshot["action_id"],
        action_name=snapshot["action_name"],
        reward=row["reward"],
        title=f"Training Episode {row['episode']} | {row.get('case_name')}",
    )


def run_live_ppo_training(episodes=25):
    trainer = PPOTrainer(BugHuntEnv)
    logs: List[dict] = []

    for ep in range(int(episodes)):
        batch = trainer.collect_episode(seed=ep)
        loss = trainer.update(batch)
        total_reward = sum(batch[3])
        row = {
            "episode": ep,
            "reward": total_reward,
            "loss": loss,
            "actions": batch[6],
            "probs": batch[5],
            "entropy": sum(batch[7]) / max(1, len(batch[7])),
            "success": any(action[0] == 4 for action in batch[6]) and total_reward > 0,
            "case_name": batch[8][0] if batch[8] else None,
            "activations": batch[9],
        }
        logs.append(row)

        df = build_dashboard_rows(logs)
        heatmap_df = case_action_heatmap(logs)
        yield (
            summarize_logs(logs),
            df,
            render_dashboard_viz(row),
            plot_reward_curve(df),
            plot_loss_curve(df),
            plot_success_curve(df),
            plot_action_distribution(df),
            plot_entropy_curve(df),
            plot_case_action_heatmap(heatmap_df),
        )
