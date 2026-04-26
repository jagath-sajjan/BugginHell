from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

from bughunt_env.environment import BugHuntObservation
from bughunt_env.ppo_agent import build_action


ACTION_TO_ID = {
    "read_file": 0,
    "run_test": 1,
    "search_symbol": 2,
    "trace_caller": 3,
    "commit_location": 4,
}

DEFAULT_BASE_URL = "https://ai.hackclub.com/proxy/v1"
DEFAULT_API_KEY = "sk-hc-v1-2de503367fa6405bbc2c54865d51247136731a52445d401bb8b2dc3390ca2e2a"
DEFAULT_MODEL = "qwen/qwen3-32b"


@dataclass
class LLMDecision:
    action: Tuple[int, Dict[str, Any]]
    raw_response: str
    prompt: str
    parse_error: Optional[str] = None


def build_llm_prompt(obs: BugHuntObservation) -> str:
    history = "\n".join(f"- {entry}" for entry in obs.history[-8:]) or "- none yet"
    file_tree = "\n".join(f"- {path}" for path in obs.file_tree)
    return f"""
You are BugHunt, a debugging agent navigating a Python codebase.

Pick exactly one tool call as JSON.

Allowed schemas:
{{"tool":"read_file","path":"file.py"}}
{{"tool":"run_test","name":"test_name"}}
{{"tool":"search_symbol","name":"symbol"}}
{{"tool":"trace_caller","fn":"function_name"}}
{{"tool":"commit_location","file":"file.py","line":12}}

Rules:
- Return JSON only.
- Use the observation to narrow the bug before committing.
- Prefer a source file over a test file when reading or committing.
- If the error names a function, search or trace that function.
- Commit only when the bug file and likely line are grounded in evidence from prior tool outputs.

Current observation:
Files:
{file_tree}

Failing test:
{obs.failing_test}

stderr:
{obs.stderr or "<empty>"}

last_tool_output:
{obs.last_tool_output or "<empty>"}

history:
{history}

steps_left: {obs.steps_left}
""".strip()


def parse_llm_action(text: str, obs: BugHuntObservation):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")

    data = json.loads(match.group(0))
    tool = data.get("tool")
    if tool == "read_file":
        source_files = [path for path in obs.file_tree if path.endswith(".py") and "test" not in path]
        fallback = source_files[0] if source_files else obs.file_tree[0]
        return (0, {"path": data.get("path", fallback)})
    if tool == "run_test":
        return (1, {"name": data.get("name", obs.failing_test)})
    if tool == "search_symbol":
        return (2, {"name": data.get("name", obs.failing_test)})
    if tool == "trace_caller":
        return (3, {"fn": data.get("fn", obs.failing_test)})
    if tool == "commit_location":
        source_files = [path for path in obs.file_tree if path.endswith(".py") and "test" not in path]
        fallback = source_files[0] if source_files else obs.file_tree[0]
        return (4, {"file": data.get("file", fallback), "line": int(data.get("line", 1))})

    raise ValueError(f"Unknown tool: {tool}")


class APIAgent:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key or os.getenv("HACKCLUB_API_KEY") or os.getenv("OPENAI_API_KEY") or DEFAULT_API_KEY,
            base_url=base_url,
            timeout=timeout,
        )

    def decide(self, obs: BugHuntObservation) -> LLMDecision:
        prompt = build_llm_prompt(obs)
        raw_text = ""
        parse_error: Optional[str] = None

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a precise debugging agent. Return one JSON tool call only."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw_text = response.choices[0].message.content or ""
            action = parse_llm_action(raw_text, obs)
        except Exception as exc:
            parse_error = str(exc)
            action = build_action(1 if obs.steps_left > 1 else 4, obs)

        return LLMDecision(action=action, raw_response=raw_text, prompt=prompt, parse_error=parse_error)

    def act(self, obs: BugHuntObservation):
        decision = self.decide(obs)
        return decision.action, decision.raw_response


class LocalLLMAgent(APIAgent):
    pass
