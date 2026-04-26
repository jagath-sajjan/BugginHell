import sys
from pathlib import Path
import random
import torch
import torch.nn as nn
import pandas as pd
import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bughunt_env import BugHuntEnv, LocalLLMAgent
from bughunt_env.ppo_agent import PPOTrainer, encode_obs, build_action
from server.ppo_live import run_live_ppo_training
from server.code_workspace import (
    DEFAULT_CODEBASE,
    parse_pasted_files,
    read_zip_codebase,
    make_file_tree,
    get_file_content,
    format_codebase_for_display,
)


ACTION_NAMES = [
    "read_file",
    "run_test",
    "search_symbol",
    "trace_caller",
    "commit_location",
]


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


MODEL_PATH = PROJECT_ROOT / "outputs" / "policy_net.pt"
policy = PolicyNet()

if MODEL_PATH.exists():
    policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    policy.eval()


def run_random():
    env = BugHuntEnv(seed=1)
    obs, _ = env.reset(seed=1)
    logs = []
    total = 0

    for _ in range(env.max_steps):
        action_id = random.randint(0, 4)
        action = build_action(action_id, obs)
        obs, reward, term, trunc, _ = env.step(action)
        total += reward
        logs.append(f"{action} -> reward={reward}")

        if term or trunc:
            break

    return "\n".join(logs) + f"\nTOTAL: {round(total, 2)}"


def run_rl():
    env = BugHuntEnv(seed=1)
    obs, _ = env.reset(seed=1)
    logs = []
    total = 0

    for _ in range(env.max_steps):
        x = encode_obs(obs)
        with torch.no_grad():
            action_id = torch.argmax(policy(x)).item()

        action = build_action(action_id, obs)
        obs, reward, term, trunc, _ = env.step(action)
        total += reward
        logs.append(f"{action} -> reward={reward}")

        if term or trunc:
            break

    return "\n".join(logs) + f"\nTOTAL: {round(total, 2)}"


def run_llm():
    env = BugHuntEnv(seed=1)
    obs, _ = env.reset(seed=1)

    try:
        agent = LocalLLMAgent()
    except Exception as exc:
        return f"LLM model could not load on this machine/Space.\n\n{exc}"

    logs = []
    total = 0

    for _ in range(env.max_steps):
        action, raw = agent.act(obs)
        obs, reward, term, trunc, _ = env.step(action)
        total += reward

        logs.append(f"RAW: {raw}")
        logs.append(f"ACTION: {action} -> reward={reward}")
        logs.append("-" * 70)

        if term or trunc:
            break

    return "\n".join(logs) + f"\nTOTAL: {round(total, 2)}"


def load_workspace(pasted_text, zip_file):
    if zip_file is not None:
        files = read_zip_codebase(zip_file)
    elif pasted_text and pasted_text.strip():
        files = parse_pasted_files(pasted_text)
    else:
        files = DEFAULT_CODEBASE

    file_names = sorted(files.keys())
    first = file_names[0]

    return (
        files,
        make_file_tree(files),
        gr.Dropdown(choices=file_names, value=first),
        get_file_content(files, first),
        format_codebase_for_display(files),
    )


def open_selected_file(files, selected_file):
    if not files:
        files = DEFAULT_CODEBASE
    return get_file_content(files, selected_file)


def run_main_live_ppo(files):
    if not files:
        files = DEFAULT_CODEBASE

    trainer = PPOTrainer(BugHuntEnv)
    logs = trainer.train(episodes=25)

    env = BugHuntEnv(seed=1)
    obs, info = env.reset(seed=1)

    code_view = get_file_content(files, sorted(files.keys())[0])
    decision_lines = []
    neural_rows = []
    reward_rows = []

    for ep, row in enumerate(logs):
        reward_rows.append(
            {
                "episode": row["episode"],
                "reward": row["reward"],
                "loss": row["loss"],
            }
        )

        if row["probs"]:
            probs = row["probs"][-1]
        else:
            probs = [0, 0, 0, 0, 0]

        neural_rows.append(
            {
                "episode": row["episode"],
                "read_file": probs[0],
                "run_test": probs[1],
                "search_symbol": probs[2],
                "trace_caller": probs[3],
                "commit_location": probs[4],
            }
        )

        decision_lines.append(f"EP {row['episode']} | R={row['reward']:.2f} | L={row['loss']:.3f}")

        for action in row["actions"]:
            action_id, params = action
            decision_lines.append(f"→ {ACTION_NAMES[action_id]}")

        decision_lines.append("-" * 70)

    neural_df = pd.DataFrame(neural_rows)
    reward_df = pd.DataFrame(reward_rows)

    return code_view, neural_df, "\n".join(decision_lines), reward_df


with gr.Blocks(css="""
body {
    background: #050505;
}
#main-title {
    text-align: center;
}
.big-card textarea {
    font-family: monospace !important;
}
""") as app:
    gr.Markdown(
        "# Live Nerd Stats Of The RL On This CodeBase",
        elem_id="main-title",
    )

    workspace_state = gr.State(DEFAULT_CODEBASE)

    with gr.Tab("Main RL Codebase Lab"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## Codebase")
                pasted = gr.Textbox(
                    label="Paste files here",
                    lines=10,
                    value="""=== cart.py ===
def calculate_total(items):
    total = 0
    for i in range(len(items) - 1):
        total += items[i]["price"]
    return total

=== tests/test_cart.py ===
from cart import calculate_total

def test_calculate_total_counts_all_items():
    items = [{"price": 10}, {"price": 20}, {"price": 30}]
    assert calculate_total(items) == 60
""",
                )
                zip_upload = gr.File(label="Or upload a .zip codebase", file_types=[".zip"])
                load_btn = gr.Button("Load Codebase")

                file_tree = gr.Textbox(label="File Tree", lines=8)
                file_select = gr.Dropdown(label="Open File", choices=[])
                code_view = gr.Code(label="Code Viewer", language="python", lines=28)

            with gr.Column(scale=1):
                gr.Markdown("## Live Neural Activity Layers Map")
                neural_map = gr.Dataframe(label="Action Probability Map")

                gr.Markdown("## Detailed Logs With Reward System And Stats")
                run_main_btn = gr.Button("Run Live RL On This Codebase")
                ppo_logs = gr.Textbox(label="RL Decision Logs", lines=18)
                reward_table = gr.Dataframe(label="Reward / Loss Stats")

        all_code = gr.Textbox(label="Full Codebase Raw View", visible=False)

        load_btn.click(
            fn=load_workspace,
            inputs=[pasted, zip_upload],
            outputs=[workspace_state, file_tree, file_select, code_view, all_code],
        )

        file_select.change(
            fn=open_selected_file,
            inputs=[workspace_state, file_select],
            outputs=[code_view],
        )

        run_main_btn.click(
            fn=run_main_live_ppo,
            inputs=[workspace_state],
            outputs=[code_view, neural_map, ppo_logs, reward_table],
        )

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

    with gr.Tab("Live RL Dashboard"):
        gr.Markdown("Train a fresh RL agent live on the BugHunt environment.")

        episodes = gr.Slider(
            minimum=5,
            maximum=100,
            value=25,
            step=5,
            label="RL Episodes",
        )

        ppo_btn = gr.Button("Run Live RL")

        with gr.Row():
            ppo_output = gr.Textbox(label="Live RL Training Log", lines=28)
            ppo_table = gr.Dataframe(label="Episode Metrics")

        with gr.Row():
            reward_plot = gr.Plot(label="Reward Curve")
            loss_plot = gr.Plot(label="Loss Curve")

        ppo_btn.click(
            fn=run_live_ppo_training,
            inputs=[episodes],
            outputs=[ppo_output, ppo_table, reward_plot, loss_plot],
        )


if __name__ == "__main__":
    app.launch()
