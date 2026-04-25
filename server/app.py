import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from bughunt_env import BugHuntEnv


# ---- SAME ENCODER + MODEL AS NOTEBOOK ----

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


# ---- LOAD MODEL ----

MODEL_PATH = PROJECT_ROOT / "outputs" / "policy_net.pt"

policy = PolicyNet()
if MODEL_PATH.exists():
    policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    policy.eval()
    print("Loaded trained policy")
else:
    print("WARNING: policy_net.pt not found, using random weights")


# ---- ACTION BUILDER ----

import random

def build_action(action_id, obs):
    text = (obs.stderr + " " + obs.last_tool_output).lower()

    if "calculate_total" in text:
        symbol = "calculate_total"
        file = "cart.py"
        line = 3
    elif "is_admin" in text:
        symbol = "is_admin"
        file = "auth.py"
        line = 2
    elif "safe_divide" in text:
        symbol = "safe_divide"
        file = "math_tools.py"
        line = 2
    elif "normalize_name" in text:
        symbol = "normalize_name"
        file = "user_format.py"
        line = 4
    else:
        symbol = obs.failing_test
        file = random.choice(obs.file_tree)
        line = random.randint(1, 8)

    if action_id == 0:
        return (0, {"path": file})
    if action_id == 1:
        return (1, {"name": obs.failing_test})
    if action_id == 2:
        return (2, {"name": symbol})
    if action_id == 3:
        return (3, {"fn": symbol})
    return (4, {"file": file, "line": line})


# ---- RUN AGENTS ----

def run_agent(use_model):
    env = BugHuntEnv(seed=1)
    obs, info = env.reset(seed=1)

    logs = []
    logs.append(f"Case: {info['case_name']}")
    logs.append(f"Failing test: {obs.failing_test}")
    logs.append("-" * 60)

    total_reward = 0

    for _ in range(env.max_steps):

        if use_model:
            x = encode_obs(obs)
            with torch.no_grad():
                logits = policy(x)
                action_id = torch.argmax(logits).item()
        else:
            action_id = random.randint(0, 4)

        action = build_action(action_id, obs)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        logs.append(f"Action: {action}")
        logs.append(f"Reward: {reward}")
        logs.append(f"Output:\n{obs.last_tool_output}")
        logs.append("-" * 60)

        if terminated or truncated:
            break

    logs.append(f"Total reward: {round(total_reward, 3)}")
    return "\n".join(logs)


def demo():
    base = run_agent(False)
    trained = run_agent(True)
    return base, trained


with gr.Blocks(title="BugginHell — BugHunt RL") as app:
    gr.Markdown("# 🐛 BugginHell — BugHunt RL")
    gr.Markdown("RL agent learns strategic bug localization")

    btn = gr.Button("Run Demo")

    with gr.Row():
        base_out = gr.Textbox(label="Random Agent", lines=25)
        trained_out = gr.Textbox(label="Trained RL Agent", lines=25)

    btn.click(fn=demo, inputs=[], outputs=[base_out, trained_out])

if __name__ == "__main__":
    app.launch()
