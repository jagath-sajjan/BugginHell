import json
import re
from typing import Optional

from bughunt_env.environment import BugHuntObservation


ACTION_TO_ID = {
    "read_file": 0,
    "run_test": 1,
    "search_symbol": 2,
    "trace_caller": 3,
    "commit_location": 4,
}


def build_llm_prompt(obs: BugHuntObservation) -> str:
    return f"""
You are BugHunt RL, a debugging agent inside a Python repository.

Your job:
Find the hidden bug using as few tool calls as possible.

Allowed tools:
{{"tool":"read_file","path":"file.py"}}
{{"tool":"run_test","name":"test_name"}}
{{"tool":"search_symbol","name":"symbol"}}
{{"tool":"trace_caller","fn":"function"}}
{{"tool":"commit_location","file":"file.py","line":3}}

Rules:
- Return ONLY valid JSON.
- Do not explain.
- Prefer search_symbol first when you see a failing test.
- Prefer trace_caller after identifying a function.
- Commit only when you know the likely bug file and line.

Files:
{obs.file_tree}

Failing test:
{obs.failing_test}

Error:
{obs.stderr}

Last tool output:
{obs.last_tool_output}

History:
{obs.history}

Steps left:
{obs.steps_left}
""".strip()


def parse_llm_action(text: str, obs: BugHuntObservation):
    try:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found")

        data = json.loads(match.group(0))
        tool = data.get("tool")

        if tool == "read_file":
            return (0, {"path": data.get("path", obs.file_tree[0])})

        if tool == "run_test":
            return (1, {"name": data.get("name", obs.failing_test)})

        if tool == "search_symbol":
            return (2, {"name": data.get("name", obs.failing_test)})

        if tool == "trace_caller":
            return (3, {"fn": data.get("fn", "")})

        if tool == "commit_location":
            return (
                4,
                {
                    "file": data.get("file", obs.file_tree[0]),
                    "line": int(data.get("line", 1)),
                },
            )

    except Exception:
        pass

    return (1, {"name": obs.failing_test})


class LocalLLMAgent:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.model.eval()

    def act(self, obs: BugHuntObservation):
        prompt = build_llm_prompt(obs)

        messages = [
            {"role": "system", "content": "You are a precise code debugging agent."},
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            input_text = prompt

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return parse_llm_action(text, obs), text
