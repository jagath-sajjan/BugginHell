import sys
from pathlib import Path
import torch
import torch.nn as nn
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from server.ppo_live import run_live_ppo_training
from bughunt_env import BugHuntEnv, LocalLLMAgent

class PolicyNet(nn.Module):
    def __init__(self, input_dim=13, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def encode_obs(obs):
    text = " ".join([
        " ".join(obs.file_tree),
        obs.failing_test,
        obs.stderr,
        obs.last_tool_output,
        str(obs.steps_left),
    ]).lower()

    features = [
        obs.steps_left / 10,
        float("failed" in text),
        float("assert" in text),
        float("zerodivision" in text),
        float("calculate_total" in text),
        float("is_admin" in text),
        float("safe_divide" in text),
        float("normalize_name" in text),
        float("cart.py" in text),
        float("auth.py" in text),
        float("math_tools.py" in text),
        float("user_format.py" in text),
        float("tests/" in text),
    ]

    return torch.tensor(features, dtype=torch.float32)


MODEL_PATH = PROJECT_ROOT / "outputs" / "policy_net.pt"
policy = PolicyNet()

if MODEL_PATH.exists():
    policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    policy.eval()

def build_action(action_id, obs):
    file = random.choice(obs.file_tree)
    line = random.randint(1, 8)

    if action_id == 0:
        return (0, {"path": file})
    if action_id == 1:
        return (1, {"name": obs.failing_test})
    if action_id == 2:
        return (2, {"name": obs.failing_test})
    if action_id == 3:
        return (3, {"fn": obs.failing_test})
    return (4, {"file": file, "line": line})

def run_random():
    env = BugHuntEnv(seed=1)
    obs, info = env.reset(seed=1)
    logs = []
    total = 0

    for _ in range(env.max_steps):
        action = build_action(random.randint(0, 4), obs)
        obs, reward, term, trunc, _ = env.step(action)
        total += reward
        logs.append(f"{action} -> {reward}")

        if term or trunc:
            break

    return "\n".join(logs) + f"\nTOTAL: {round(total,2)}"


def run_rl():
    env = BugHuntEnv(seed=1)
    obs, info = env.reset(seed=1)
    logs = []
    total = 0

    for _ in range(env.max_steps):
        x = encode_obs(obs)
        with torch.no_grad():
            action_id = torch.argmax(policy(x)).item()

        action = build_action(action_id, obs)
        obs, reward, term, trunc, _ = env.step(action)
        total += reward
        logs.append(f"{action} -> {reward}")

        if term or trunc:
            break

    return "\n".join(logs) + f"\nTOTAL: {round(total,2)}"


def run_llm():
    env = BugHuntEnv(seed=1)
    obs, info = env.reset(seed=1)
    agent = LocalLLMAgent()

    logs = []
    total = 0

    for _ in range(env.max_steps):
        action, raw = agent.act(obs)
        obs, reward, term, trunc, _ = env.step(action)
        total += reward

        logs.append(f"RAW: {raw}")
        logs.append(f"ACTION: {action} -> {reward}")

        if term or trunc:
            break

    return "\n".join(logs) + f"\nTOTAL: {round(total,2)}"

with gr.Blocks() as app:
    gr.Markdown("# BugginHell — RL Bug Hunter")
    gr.Markdown("A PPO-powered environment where agents learn strategic bug localization.")

    with gr.Tab("Agent Comparison"):
        btn = gr.Button("Run Agent Comparison")

        with gr.Row():
            out1 = gr.Textbox(label="Random Agent", lines=20)
            out2 = gr.Textbox(label="RL Agent", lines=20)
            out3 = gr.Textbox(label="LLM Agent", lines=20)

        btn.click(
            fn=lambda: (run_random(), run_rl(), run_llm()),
            outputs=[out1, out2, out3],
        )

    with gr.Tab("Live PPO Training"):
        gr.Markdown("Train a fresh PPO agent live on the BugHunt environment.")

        episodes = gr.Slider(
            minimum=5,
            maximum=100,
            value=25,
            step=5,
            label="PPO Episodes",
        )

        ppo_btn = gr.Button("Run Live PPO")
        ppo_output = gr.Textbox(label="Live PPO Training Log", lines=30)

        ppo_btn.click(
            fn=run_live_ppo_training,
            inputs=[episodes],
            outputs=[ppo_output],
        )

if __name__ == "__main__":
    app.launch()
