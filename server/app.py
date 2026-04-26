from __future__ import annotations

import difflib
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, str(PROJECT_ROOT))

from bughunt_env import APIAgent, BugHuntEnv, ExternalBugHuntEnv, RandomAgent
from bughunt_env.fixtures import get_cases
from bughunt_env.ppo_agent import ACTION_NAMES, FEATURE_NAMES, PPOTrainer, build_action
from server.code_workspace import format_codebase_for_display, get_file_content, make_file_tree
from server.neural_viz import render_neural_network_svg
from server.ppo_live import run_live_ppo_training
from server.source_loader import WorkspaceSpec, load_source


CASE_MAP = {case.name: case for case in get_cases()}
REPLAY_PPO_TRAINER: Optional[PPOTrainer] = None


def get_replay_trainer() -> PPOTrainer:
    global REPLAY_PPO_TRAINER
    if REPLAY_PPO_TRAINER is None:
        trainer = PPOTrainer(BugHuntEnv)
        trainer.train(episodes=80)
        REPLAY_PPO_TRAINER = trainer
    return REPLAY_PPO_TRAINER


def observation_markdown(obs, *, title: str) -> str:
    history = "\n".join(f"- {item}" for item in obs.history[-6:]) or "- none"
    file_tree = "\n".join(f"- {item}" for item in obs.file_tree)
    return (
        f"### {title}\n"
        f"**steps_left**: `{obs.steps_left}`\n\n"
        f"**failing_test**: `{obs.failing_test}`\n\n"
        f"**file_tree**\n```\n{file_tree}\n```\n"
        f"**stderr**\n```text\n{obs.stderr or '<empty>'}\n```\n"
        f"**last_tool_output**\n```text\n{obs.last_tool_output or '<empty>'}\n```\n"
        f"**history**\n```text\n{history}\n```"
    )


def case_overview_markdown(case_name: str) -> str:
    case = CASE_MAP[case_name]
    return (
        f"## Case: `{case.name}`\n"
        f"**Bug summary**: {case.bug_summary}\n\n"
        f"**Ground truth**: `{case.bug_file}:{case.bug_line}`\n\n"
        f"**Failing test**: `{case.failing_test}`\n\n"
        f"**Files**\n```text\n" + "\n".join(case.file_tree) + "\n```"
    )


def case_source_bundle(case_name: str) -> str:
    return format_codebase_for_display(CASE_MAP[case_name].files)


def workspace_spec_to_state(spec: WorkspaceSpec) -> Dict[str, Any]:
    return {
        "label": spec.label,
        "source_type": spec.source_type,
        "files": spec.files,
        "origin": spec.origin,
    }


def workspace_summary_markdown(spec_state: Dict[str, Any]) -> str:
    files = spec_state["files"]
    file_names = sorted(files.keys())
    py_files = [name for name in file_names if name.endswith(".py")]
    test_files = [name for name in py_files if "test" in name.lower()]
    html_files = [name for name in file_names if name.endswith(".html")]
    return (
        f"## Extracted Workspace: `{spec_state['label']}`\n"
        f"**Source type**: `{spec_state['source_type']}`\n\n"
        f"**Origin**: `{spec_state['origin']}`\n\n"
        f"**Files**: `{len(file_names)}` total, `{len(py_files)}` python, `{len(test_files)}` tests, `{len(html_files)}` html\n\n"
        f"**Top files**\n```text\n" + "\n".join(file_names[:40]) + "\n```"
    )


def action_to_text(action: Tuple[int, Dict[str, Any]]) -> str:
    action_id, params = action
    return f"{ACTION_NAMES[action_id]} {params}"


def top_feature_summary(features: List[float], feature_names: List[str], limit: int = 5) -> str:
    indexed = sorted(zip(feature_names, features), key=lambda item: abs(item[1]), reverse=True)
    lines = [f"- `{name}` = `{value:.3f}`" for name, value in indexed[:limit]]
    return "\n".join(lines) or "- none"


def reward_breakdown_plot(reward_breakdown: Optional[Dict[str, Any]]):
    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    if not reward_breakdown:
        ax.text(0.5, 0.5, "Reward anatomy appears after commit_location.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    labels = ["step_penalty", "premature", "file", "line", "efficiency"]
    values = [
        reward_breakdown["base_step_penalty"],
        reward_breakdown.get("premature_commit_penalty", 0.0),
        reward_breakdown["file_score"],
        reward_breakdown["line_score"],
        reward_breakdown["efficiency_bonus_score"],
    ]
    colors = ["#d62828" if value < 0 else "#2a9d8f" for value in values]
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#1d3557", linewidth=1)
    total = reward_breakdown["reward"]
    ax.set_title(f"Reward anatomy | total={total:.2f}")
    ax.set_xlabel("Contribution")
    for idx, value in enumerate(values):
        ax.text(value + (0.03 if value >= 0 else -0.18), idx, f"{value:+.2f}", va="center")
    fig.tight_layout()
    return fig


def observation_diff_html(before, after) -> str:
    before_text = (
        f"steps_left: {before.steps_left}\n"
        f"last_tool_output:\n{before.last_tool_output or '<empty>'}\n\n"
        f"history:\n" + ("\n".join(before.history) or "<empty>")
    )
    after_text = (
        f"steps_left: {after.steps_left}\n"
        f"last_tool_output:\n{after.last_tool_output or '<empty>'}\n\n"
        f"history:\n" + ("\n".join(after.history) or "<empty>")
    )
    diff = difflib.HtmlDiff(wrapcolumn=80).make_table(
        before_text.splitlines(),
        after_text.splitlines(),
        fromdesc="Before",
        todesc="After",
        context=True,
        numlines=2,
    )
    return (
        "<div style='background:#08111f;border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:12px;overflow:auto;'>"
        "<div style='font:600 13px monospace;color:#e6eefb;margin-bottom:8px;'>Observation diff</div>"
        f"{diff}</div>"
    )


def blank_diff_html() -> str:
    return (
        "<div style='background:#08111f;border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:12px;'>"
        "<div style='font:600 13px monospace;color:#e6eefb;'>Observation diff</div>"
        "<div style='font:13px monospace;color:#9fb3c8;margin-top:8px;'>No action has executed yet.</div>"
        "</div>"
    )


def blank_neural_note(message: str, *, title: str = "Neural Activity", compact: bool = False) -> str:
    return (
        f"<div class='viz-placeholder{' viz-placeholder-compact' if compact else ''}'>"
        f"<div class='viz-placeholder-title'>{title}</div>"
        f"<div class='viz-placeholder-body'>{message}</div>"
        "</div>"
    )


def make_agent_session(agent_kind: str, case_name: str) -> Dict[str, Any]:
    env = BugHuntEnv(seed=1)
    obs, info = env.reset(seed=1, options={"case_name": case_name})
    if agent_kind == "random":
        helper = RandomAgent()
    elif agent_kind == "api":
        helper = APIAgent()
    else:
        helper = get_replay_trainer()
    session = {
        "kind": agent_kind,
        "env": env,
        "obs": obs,
        "info": info,
        "helper": helper,
        "done": False,
        "step_index": 0,
        "last_transition": None,
        "next_decision": None,
    }
    session["next_decision"] = plan_decision(session)
    return session


def make_external_agent_session(agent_kind: str, spec_state: Dict[str, Any]) -> Dict[str, Any]:
    env = ExternalBugHuntEnv(
        files=spec_state["files"],
        label=spec_state["label"],
        source_type=spec_state["source_type"],
    )
    obs, info = env.reset(seed=1)
    if agent_kind == "random":
        helper = RandomAgent()
    elif agent_kind == "api":
        helper = APIAgent()
    else:
        helper = get_replay_trainer()
    session = {
        "kind": agent_kind,
        "env": env,
        "obs": obs,
        "info": info,
        "helper": helper,
        "done": False,
        "step_index": 0,
        "last_transition": None,
        "next_decision": None,
    }
    session["next_decision"] = plan_decision(session)
    return session


def plan_decision(session: Dict[str, Any]) -> Dict[str, Any]:
    obs = session["obs"]
    if session["done"]:
        label = {"ppo": "PPO Agent", "random": "Random Agent", "api": "API LLM Agent"}[session["kind"]]
        return {
            "action": None,
            "reason": "Episode finished.",
            "viz_html": blank_neural_note("Episode finished.", title=label, compact=session["kind"] != "ppo"),
            "inspection": None,
        }

    if session["kind"] == "random":
        action = session["helper"].act(obs)
        return {
            "action": action,
            "reason": "Uniform random baseline. No stateful policy; this step is sampled at random.",
            "viz_html": blank_neural_note(
                "Random baseline has no internal neural activations to render.",
                title="Random Agent",
                compact=True,
            ),
            "inspection": None,
        }

    if session["kind"] == "api":
        decision = session["helper"].decide(obs)
        reason = "API LLM decision.\n"
        if decision.parse_error:
            reason += f"\nFallback triggered: `{decision.parse_error}`"
        reason += f"\n\nRaw response:\n```json\n{decision.raw_response or '<empty>'}\n```"
        return {
            "action": decision.action,
            "reason": reason,
            "viz_html": blank_neural_note(
                "Qwen runs remotely through the Hack Club API; token-level activations are not exposed by the endpoint.",
                title="API LLM Agent",
                compact=True,
            ),
            "inspection": None,
        }

    trainer = session["helper"]
    inspection = trainer.model.inspect(obs)
    action_id = inspection["suggested_action_id"]
    action = build_action(action_id, obs)
    reason = (
        f"Top policy action: `{ACTION_NAMES[action_id]}`\n\n"
        f"Top features:\n{top_feature_summary(inspection['features'], inspection['feature_names'])}\n\n"
        f"Feature vector:\n```text\n"
        + ", ".join(f"{name}={value:.3f}" for name, value in zip(inspection["feature_names"], inspection["features"]))
        + "\n```"
    )
    viz_html = render_neural_network_svg(
        features=inspection["features"],
        h1=inspection["h1"],
        h2=inspection["h2"],
        logits=inspection["logits"],
        probs=inspection["probs"],
        c1=inspection["c1"],
        c2=inspection["c2"],
        c3=inspection["c3"],
        value_estimate=inspection["value"],
        step_index=session["step_index"],
        episode_index=0,
        feature_names=inspection["feature_names"],
        action_id=action_id,
        action_name=ACTION_NAMES[action_id],
        title="Live PPO Policy Snapshot",
    )
    return {
        "action": action,
        "reason": reason,
        "viz_html": viz_html,
        "inspection": inspection,
    }


def step_agent_session(session: Dict[str, Any]) -> Dict[str, Any]:
    if session["done"]:
        return session

    before = session["obs"]
    decision = session["next_decision"] or plan_decision(session)
    action = decision["action"]
    after, reward, terminated, truncated, info = session["env"].step(action)
    session["obs"] = after
    session["info"] = info
    session["done"] = bool(terminated or truncated)
    session["step_index"] += 1
    session["last_transition"] = {
        "before": before,
        "after": after,
        "action": action,
        "reward": reward,
        "info": info,
        "decision": decision,
    }
    session["next_decision"] = plan_decision(session)
    return session


def render_agent_outputs(session: Dict[str, Any]) -> Tuple[str, str, str, Any, str]:
    label = {"ppo": "PPO agent", "random": "Random agent", "api": "API LLM agent"}[session["kind"]]
    obs_md = observation_markdown(session["obs"], title=label)

    last = session["last_transition"]
    next_decision = session["next_decision"]
    if last:
        reward_breakdown = last["info"].get("reward_breakdown")
        action_md = (
            f"### Action trace\n"
            f"**executed**: `{action_to_text(last['action'])}`\n\n"
            f"**reward**: `{last['reward']:.3f}`\n\n"
            f"**reasoning**\n{last['decision']['reason']}\n\n"
            f"**next**: `{action_to_text(next_decision['action']) if next_decision['action'] else 'episode finished'}`"
        )
        diff_html = observation_diff_html(last["before"], last["after"])
        reward_plot = reward_breakdown_plot(reward_breakdown)
    else:
        action_md = (
            f"### Action preview\n"
            f"**next**: `{action_to_text(next_decision['action']) if next_decision['action'] else 'episode finished'}`\n\n"
            f"**reasoning**\n{next_decision['reason']}"
        )
        diff_html = blank_diff_html()
        reward_plot = reward_breakdown_plot(None)

    viz_html = next_decision["viz_html"]
    return obs_md, action_md, viz_html, reward_plot, diff_html


def render_live_outputs(state: Dict[str, Any]):
    outputs: List[Any] = [state, case_overview_markdown(state["case_name"]), case_source_bundle(state["case_name"])]
    for key in ["ppo", "random", "api"]:
        outputs.extend(render_agent_outputs(state["agents"][key]))
    return outputs


def render_external_outputs(state: Dict[str, Any]):
    spec_state = state["spec"]
    files = spec_state["files"]
    file_names = sorted(files.keys())
    first = file_names[0] if file_names else ""
    outputs: List[Any] = [
        state,
        spec_state,
        workspace_summary_markdown(spec_state),
        make_file_tree(files),
        gr.Dropdown(choices=file_names, value=first),
        get_file_content(files, first) if first else "",
        format_codebase_for_display(files),
    ]
    for key in ["ppo", "random", "api"]:
        outputs.extend(render_agent_outputs(state["agents"][key]))
    return outputs


def init_live_episode(case_name: str):
    state = {
        "case_name": case_name,
        "agents": {
            "ppo": make_agent_session("ppo", case_name),
            "random": make_agent_session("random", case_name),
            "api": make_agent_session("api", case_name),
        },
    }
    return render_live_outputs(state)


def step_live_episode(state: Optional[Dict[str, Any]], case_name: str):
    if not state or state.get("case_name") != case_name:
        state = {
            "case_name": case_name,
            "agents": {
                "ppo": make_agent_session("ppo", case_name),
                "random": make_agent_session("random", case_name),
                "api": make_agent_session("api", case_name),
            },
        }
    for key in ["ppo", "random", "api"]:
        state["agents"][key] = step_agent_session(state["agents"][key])
    return render_live_outputs(state)


def run_full_episode(state: Optional[Dict[str, Any]], case_name: str):
    if not state or state.get("case_name") != case_name:
        state = init_live_episode(case_name)[0]
    active = True
    while active:
        active = False
        for key in ["ppo", "random", "api"]:
            if not state["agents"][key]["done"]:
                state["agents"][key] = step_agent_session(state["agents"][key])
                active = True
        yield render_live_outputs(state)
        if active:
            time.sleep(0.5)


def run_text_episode(agent_kind: str, case_name: str) -> str:
    session = make_agent_session(agent_kind, case_name)
    lines = [f"Case: {case_name}", f"Agent: {agent_kind}"]
    while not session["done"]:
        session = step_agent_session(session)
        step = session["last_transition"]
        lines.append(
            f"step={session['step_index']} action={action_to_text(step['action'])} "
            f"reward={step['reward']:.3f} done={session['done']}"
        )
        lines.append(step["after"].last_tool_output)
        lines.append("-" * 88)
    return "\n".join(lines)


def load_external_workspace(source_input: str, zip_file, pasted_text: str):
    spec = load_source(source_input=source_input, zip_file=zip_file, pasted_text=pasted_text)
    spec_state = workspace_spec_to_state(spec)
    state = {
        "spec": spec_state,
        "agents": {
            "ppo": make_external_agent_session("ppo", spec_state),
            "random": make_external_agent_session("random", spec_state),
            "api": make_external_agent_session("api", spec_state),
        },
    }
    return render_external_outputs(state)


def external_open_selected_file(spec_state: Optional[Dict[str, Any]], selected_file: str):
    if not spec_state:
        return ""
    return get_file_content(spec_state["files"], selected_file)


def step_external_workspace(state: Optional[Dict[str, Any]], spec_state: Optional[Dict[str, Any]]):
    if not state:
        if not spec_state:
            raise gr.Error("Load a repo, site, local path, zip, or pasted project first.")
        state = {
            "spec": spec_state,
            "agents": {
                "ppo": make_external_agent_session("ppo", spec_state),
                "random": make_external_agent_session("random", spec_state),
                "api": make_external_agent_session("api", spec_state),
            },
        }
    for key in ["ppo", "random", "api"]:
        if not state["agents"][key]["done"]:
            state["agents"][key] = step_agent_session(state["agents"][key])
    return render_external_outputs(state)


def run_external_full_episode(state: Optional[Dict[str, Any]], spec_state: Optional[Dict[str, Any]]):
    if not state:
        if not spec_state:
            raise gr.Error("Load a repo, site, local path, zip, or pasted project first.")
        state = {
            "spec": spec_state,
            "agents": {
                "ppo": make_external_agent_session("ppo", spec_state),
                "random": make_external_agent_session("random", spec_state),
                "api": make_external_agent_session("api", spec_state),
            },
        }
    active = True
    while active:
        active = False
        for key in ["ppo", "random", "api"]:
            if not state["agents"][key]["done"]:
                state["agents"][key] = step_agent_session(state["agents"][key])
                active = True
        yield render_external_outputs(state)
        if active:
            time.sleep(0.5)


CUSTOM_CSS = """
body { background: radial-gradient(circle at top, #132238 0%, #060a11 45%, #020407 100%); }
.gradio-container { max-width: 1700px !important; }
.panel h3, .panel p, .panel li, .panel code { color: #e6eefb; }
.viz-panel { min-height: 610px; }
.viz-panel iframe { width: 100% !important; height: 610px !important; display: block; }
.viz-panel-hero iframe { width: 100% !important; height: 640px !important; display: block; }
.viz-placeholder {
  min-height: 610px;
  height: 610px;
  background: linear-gradient(180deg,#08111f 0%,#04070d 100%);
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 20px;
  padding: 22px;
  color: #dbe7f6;
  font: 14px ui-monospace, SFMono-Regular, Menlo, monospace;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.viz-placeholder-title {
  font-size: 15px;
  color: #f3f7fd;
}
.viz-placeholder-body {
  color: #aabbd2;
  line-height: 1.6;
  max-width: 28ch;
}
.viz-placeholder-compact {
  min-height: 220px;
  height: 220px;
}
table.diff { font-family: monospace; font-size: 12px; color: #dbe7f6; width: 100%; }
table.diff td, table.diff th { padding: 4px 6px; }
.diff_add { background: rgba(42,157,143,0.24); }
.diff_sub { background: rgba(214,40,40,0.24); }

.panel pre {
    max-height: 320px;
    overflow-y: auto;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    background: rgba(0,0,0,0.15) !important;
    scrollbar-width: thin;
}
.panel pre::-webkit-scrollbar { width: 6px; }
.panel pre::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
"""


with gr.Blocks() as app:
    gr.Markdown("# BugginHell Naked RL Lab")

    live_state = gr.State(None)
    external_live_state = gr.State(None)
    external_source_state = gr.State(None)

    with gr.Tab("Live Episode"):
        with gr.Row():
            case_dropdown = gr.Dropdown(
                choices=sorted(CASE_MAP.keys()),
                value=sorted(CASE_MAP.keys())[0],
                label="Bug case",
            )
            init_btn = gr.Button("Load Case")
            step_btn = gr.Button("Step →")
            run_btn = gr.Button("Run Full Episode")

        case_overview = gr.Markdown()
        case_code = gr.Code(label="Fixture source bundle", language="python", lines=18)

        with gr.Row():
            ppo_obs = gr.Markdown(elem_classes=["panel"])
            rand_obs = gr.Markdown(elem_classes=["panel"])
            api_obs = gr.Markdown(elem_classes=["panel"])

        with gr.Row():
            ppo_action = gr.Markdown(elem_classes=["panel"])
            rand_action = gr.Markdown(elem_classes=["panel"])
            api_action = gr.Markdown(elem_classes=["panel"])

        with gr.Row():
            ppo_viz = gr.HTML(elem_classes=["viz-panel", "viz-panel-hero"])

        with gr.Row():
            rand_viz = gr.HTML(elem_classes=["viz-panel"])
            api_viz = gr.HTML(elem_classes=["viz-panel"])

        with gr.Row():
            ppo_reward = gr.Plot(label="PPO reward anatomy")
            rand_reward = gr.Plot(label="Random reward anatomy")
            api_reward = gr.Plot(label="API LLM reward anatomy")

        with gr.Row():
            ppo_diff = gr.HTML()
            rand_diff = gr.HTML()
            api_diff = gr.HTML()

        live_outputs = [
            live_state,
            case_overview,
            case_code,
            ppo_obs,
            ppo_action,
            ppo_viz,
            ppo_reward,
            ppo_diff,
            rand_obs,
            rand_action,
            rand_viz,
            rand_reward,
            rand_diff,
            api_obs,
            api_action,
            api_viz,
            api_reward,
            api_diff,
        ]

        init_btn.click(fn=init_live_episode, inputs=[case_dropdown], outputs=live_outputs)
        step_btn.click(fn=step_live_episode, inputs=[live_state, case_dropdown], outputs=live_outputs)
        run_btn.click(fn=run_full_episode, inputs=[live_state, case_dropdown], outputs=live_outputs)

    with gr.Tab("Repo / Site RL"):
        gr.Markdown("Load a GitHub repo, git URL, local project path, site URL, uploaded zip, or pasted files, then run the live agents on the extracted workspace.")

        with gr.Row():
            source_input = gr.Textbox(
                label="Repo / site / local path",
                placeholder="https://github.com/owner/repo  |  https://example.com  |  /path/to/project",
            )
            source_zip = gr.File(label="Or upload a .zip project", file_types=[".zip"])
            load_source_btn = gr.Button("Load Source")

        source_paste = gr.Textbox(
            label="Or paste files",
            lines=8,
            placeholder="=== app.py ===\nprint('hello')\n",
        )

        external_summary = gr.Markdown()
        with gr.Row():
            external_file_tree = gr.Textbox(label="Extracted file tree", lines=14)
            with gr.Column():
                external_file_select = gr.Dropdown(label="Open extracted file", choices=[])
                external_code_view = gr.Code(label="Extracted code viewer", language="python", lines=22)

        external_bundle = gr.Textbox(label="Full extracted bundle", lines=8, visible=False)

        with gr.Row():
            external_step_btn = gr.Button("Step Extracted Episode →")
            external_run_btn = gr.Button("Run Full Extracted Episode")

        with gr.Row():
            ext_ppo_obs = gr.Markdown(elem_classes=["panel"])
            ext_rand_obs = gr.Markdown(elem_classes=["panel"])
            ext_api_obs = gr.Markdown(elem_classes=["panel"])

        with gr.Row():
            ext_ppo_action = gr.Markdown(elem_classes=["panel"])
            ext_rand_action = gr.Markdown(elem_classes=["panel"])
            ext_api_action = gr.Markdown(elem_classes=["panel"])

        with gr.Row():
            ext_ppo_viz = gr.HTML(elem_classes=["viz-panel", "viz-panel-hero"])

        with gr.Row():
            ext_rand_viz = gr.HTML(elem_classes=["viz-panel"])
            ext_api_viz = gr.HTML(elem_classes=["viz-panel"])

        with gr.Row():
            ext_ppo_reward = gr.Plot(label="PPO exploration reward")
            ext_rand_reward = gr.Plot(label="Random exploration reward")
            ext_api_reward = gr.Plot(label="API LLM exploration reward")

        with gr.Row():
            ext_ppo_diff = gr.HTML()
            ext_rand_diff = gr.HTML()
            ext_api_diff = gr.HTML()

        external_outputs = [
            external_live_state,
            external_source_state,
            external_summary,
            external_file_tree,
            external_file_select,
            external_code_view,
            external_bundle,
            ext_ppo_obs,
            ext_ppo_action,
            ext_ppo_viz,
            ext_ppo_reward,
            ext_ppo_diff,
            ext_rand_obs,
            ext_rand_action,
            ext_rand_viz,
            ext_rand_reward,
            ext_rand_diff,
            ext_api_obs,
            ext_api_action,
            ext_api_viz,
            ext_api_reward,
            ext_api_diff,
        ]

        load_source_btn.click(
            fn=load_external_workspace,
            inputs=[source_input, source_zip, source_paste],
            outputs=external_outputs,
        )
        external_file_select.change(
            fn=external_open_selected_file,
            inputs=[external_source_state, external_file_select],
            outputs=[external_code_view],
        )
        external_step_btn.click(
            fn=step_external_workspace,
            inputs=[external_live_state, external_source_state],
            outputs=external_outputs,
        )
        external_run_btn.click(
            fn=run_external_full_episode,
            inputs=[external_live_state, external_source_state],
            outputs=external_outputs,
        )

    with gr.Tab("Live RL Dashboard"):
        gr.Markdown("Train PPO live and stream real metrics after each episode.")
        episodes = gr.Slider(minimum=5, maximum=100, value=25, step=5, label="Episodes")
        ppo_btn = gr.Button("Run Live PPO")

        training_log = gr.Textbox(label="Training log", lines=18)
        metrics_table = gr.Dataframe(label="Episode metrics")
        dashboard_viz = gr.HTML(label="Live neural activity")

        with gr.Row():
            reward_plot = gr.Plot(label="Reward curve")
            loss_plot = gr.Plot(label="Loss curve")

        with gr.Row():
            success_plot = gr.Plot(label="Rolling success rate")
            action_plot = gr.Plot(label="Action distribution")

        with gr.Row():
            entropy_plot = gr.Plot(label="Policy entropy")
            heatmap_plot = gr.Plot(label="Case x action heatmap")

        ppo_btn.click(
            fn=run_live_ppo_training,
            inputs=[episodes],
            outputs=[
                training_log,
                metrics_table,
                dashboard_viz,
                reward_plot,
                loss_plot,
                success_plot,
                action_plot,
                entropy_plot,
                heatmap_plot,
            ],
        )

    with gr.Tab("Agent Comparison"):
        compare_case = gr.Dropdown(
            choices=sorted(CASE_MAP.keys()),
            value=sorted(CASE_MAP.keys())[0],
            label="Bug case",
        )
        compare_btn = gr.Button("Run Full Text Comparison")
        with gr.Row():
            random_text = gr.Textbox(label="Random", lines=20)
            ppo_text = gr.Textbox(label="PPO", lines=20)
            api_text = gr.Textbox(label="API LLM", lines=20)

        compare_btn.click(
            fn=lambda case_name: (
                run_text_episode("random", case_name),
                run_text_episode("ppo", case_name),
                run_text_episode("api", case_name),
            ),
            inputs=[compare_case],
            outputs=[random_text, ppo_text, api_text],
        )

app.queue()
app.css = CUSTOM_CSS


if __name__ == "__main__":
    app.launch()
