import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from bughunt_env import BugHuntEnv


def run_agent(agent_type):
    env = BugHuntEnv(seed=1)
    obs, info = env.reset(seed=1)

    logs = []
    logs.append(f"Case: {info['case_name']}")
    logs.append(f"Files: {obs.file_tree}")
    logs.append(f"Failing test: {obs.failing_test}")
    logs.append(f"Error:\n{obs.stderr}")
    logs.append("-" * 60)

    if agent_type == "Base Agent":
        actions = [
            (0, {"path": "README.md"}),
            (0, {"path": "utils.py"}),
            (1, {"name": obs.failing_test}),
            (4, {"file": "README.md", "line": 1}),
        ]
    else:
        actions = [
            (2, {"name": obs.failing_test}),
            (3, {"fn": "calculate_total"}),
            (0, {"path": "cart.py"}),
            (4, {"file": "cart.py", "line": 3}),
        ]

    total_reward = 0

    for action in actions:
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
    base = run_agent("Base Agent")
    trained = run_agent("Trained Agent")
    return base, trained


with gr.Blocks(title="BugginHell — BugHunt RL") as app:
    gr.Markdown("# BugginHell — BugHunt RL")
    gr.Markdown("A reinforcement learning environment where agents learn strategic bug localization.")

    run_btn = gr.Button("Run Demo")

    with gr.Row():
        base_output = gr.Textbox(label="Untrained Base Agent", lines=24)
        trained_output = gr.Textbox(label="Strategic RL Agent", lines=24)

    run_btn.click(fn=demo, inputs=[], outputs=[base_output, trained_output])

app.launch()
