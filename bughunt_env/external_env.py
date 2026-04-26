from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces

from bughunt_env.environment import BugHuntObservation
from bughunt_env.reward import RewardBreakdown


@dataclass
class ExternalState:
    label: str
    files: Dict[str, str]
    file_tree: List[str]
    stderr: str
    failing_test: str
    steps_used: int = 0
    max_steps: int = 12
    last_tool_output: str = ""
    history: List[str] = field(default_factory=list)
    done: bool = False


class ExternalBugHuntEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, *, files: Dict[str, str], label: str, source_type: str, max_steps: int = 12):
        super().__init__()
        self.files = dict(files)
        self.label = label
        self.source_type = source_type
        self.max_steps = max_steps
        self._state: Optional[ExternalState] = None
        self._workspace_dir: Optional[tempfile.TemporaryDirectory[str]] = None

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({"steps_left": spaces.Discrete(max_steps + 1)})

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        file_tree = sorted(self.files.keys())
        failing_test = self._infer_test_target(file_tree)
        stderr = self._workspace_summary(file_tree)
        self._materialize_workspace()
        self._state = ExternalState(
            label=self.label,
            files=self.files,
            file_tree=file_tree,
            stderr=stderr,
            failing_test=failing_test,
            max_steps=self.max_steps,
        )
        return self._obs(), self._info()

    def step(self, action):
        self._require_reset()
        if self._state.done:
            return self._obs(), 0.0, True, False, self._info()

        action_id, params = self._parse_action(action)
        self._state.steps_used += 1

        if action_id == 0:
            output = self._read_file(params.get("path", ""))
            reward = -0.05
        elif action_id == 1:
            output = self._run_test(params.get("name", self._state.failing_test))
            reward = 0.1 if "FAILED" in output or "passed" in output.lower() or "collected" in output.lower() else -0.05
        elif action_id == 2:
            output = self._search_symbol(params.get("name", ""))
            reward = 0.15 if "found" in output.lower() else -0.05
        elif action_id == 3:
            output = self._trace_caller(params.get("fn", ""))
            reward = 0.15 if "references" in output.lower() else -0.05
        else:
            output, reward_breakdown = self._score_commit(params.get("file", ""), int(params.get("line", 1)))
            self._state.done = True
            self._state.last_tool_output = output
            self._state.history.append(self._format_history(action_id, params, output, reward_breakdown.reward))
            return self._obs(), reward_breakdown.reward, True, False, self._info(reward_breakdown)

        terminated = False
        truncated = False
        if self._state.steps_used >= self.max_steps:
            output += "\nBudget exhausted. Exploration episode stopped."
            truncated = True
            self._state.done = True

        self._state.last_tool_output = output
        self._state.history.append(self._format_history(action_id, params, output, reward))
        return self._obs(), reward, terminated, truncated, self._info()

    def close(self):
        if self._workspace_dir is not None:
            self._workspace_dir.cleanup()
            self._workspace_dir = None

    def _materialize_workspace(self):
        if self._workspace_dir is not None:
            self._workspace_dir.cleanup()
        self._workspace_dir = tempfile.TemporaryDirectory(prefix="bugginhell_workspace_")
        root = Path(self._workspace_dir.name)
        for rel_path, content in self.files.items():
            target = root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    def _infer_test_target(self, file_tree: List[str]) -> str:
        test_files = [path for path in file_tree if "test" in path.lower() and path.endswith(".py")]
        if test_files:
            return Path(test_files[0]).stem
        return Path(file_tree[0]).stem if file_tree else "workspace"

    def _workspace_summary(self, file_tree: List[str]) -> str:
        py_files = [path for path in file_tree if path.endswith(".py")]
        test_files = [path for path in py_files if "test" in path.lower()]
        html_files = [path for path in file_tree if path.endswith(".html")]
        return (
            f"EXTRACTED WORKSPACE MODE\n"
            f"source={self.source_type}\n"
            f"files={len(file_tree)} python_files={len(py_files)} test_candidates={len(test_files)} html_pages={len(html_files)}\n"
            f"target_hint={self._infer_test_target(file_tree)}"
        )

    def _read_file(self, path: str) -> str:
        if path not in self.files:
            return f"File not found: {path}"
        content = self.files[path]
        lines = content.splitlines()
        formatted = "\n".join(f"{i+1:3} | {line}" for i, line in enumerate(lines))
        return f"--- {path} ({len(lines)} lines) ---\n{formatted}"

    def _run_test(self, name: str) -> str:
        if self._workspace_dir is None:
            return "Workspace not materialized."
        test_files = [path for path in self.files if "test" in path.lower() and path.endswith(".py")]
        if not test_files:
            return "No Python test files discovered in extracted workspace."
        try:
            result = subprocess.run(
                ["pytest", "-q"],
                cwd=self._workspace_dir.name,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=20,
            )
            output = result.stdout.strip() or f"pytest exited with code {result.returncode}"
            return output[:8000]
        except Exception as exc:
            return f"Test execution failed in extracted workspace: {exc}"

    def _search_symbol(self, name: str) -> str:
        if not name:
            return "Empty symbol search."
        hits: List[str] = []
        pattern = re.compile(rf"\b{re.escape(name)}\b")
        for path, content in self.files.items():
            for line_no, line in enumerate(content.splitlines(), start=1):
                if pattern.search(line):
                    hits.append(f"{path}:{line_no}: {line.strip()[:140]}")
                    if len(hits) >= 12:
                        return "Symbol found:\n" + "\n".join(hits)
        if hits:
            return "Symbol found:\n" + "\n".join(hits)
        return f"Symbol `{name}` not found."

    def _trace_caller(self, fn: str) -> str:
        if not fn:
            return "Empty caller trace."
        hits: List[str] = []
        call_pattern = re.compile(rf"\b{re.escape(fn)}\s*\(")
        for path, content in self.files.items():
            for line_no, line in enumerate(content.splitlines(), start=1):
                if line.strip().startswith(f"def {fn}("):
                    continue
                if call_pattern.search(line):
                    hits.append(f"{path}:{line_no}: {line.strip()[:140]}")
                    if len(hits) >= 12:
                        return "References:\n" + "\n".join(hits)
        if hits:
            return "References:\n" + "\n".join(hits)
        return f"No references found for `{fn}`."

    def _score_commit(self, committed_file: str, committed_line: int) -> Tuple[str, RewardBreakdown]:
        file_exists = committed_file in self.files
        line_count = len(self.files.get(committed_file, "").splitlines()) if file_exists else 0
        valid_line = file_exists and 1 <= committed_line <= max(1, line_count)
        evidence_sufficient = len(self._state.history) >= 2
        distinct_actions = len({entry.split("(")[0] for entry in self._state.history})

        base_step_penalty = round(-0.05 * self._state.steps_used, 3)
        premature_commit_penalty = 0.0 if evidence_sufficient else -0.6
        file_score = 0.45 if file_exists else -0.8
        line_score = 0.25 if valid_line else -0.25
        efficiency_bonus_score = 0.25 if evidence_sufficient and distinct_actions >= 2 else 0.0
        reward = round(base_step_penalty + premature_commit_penalty + file_score + line_score + efficiency_bonus_score, 3)
        reason = (
            f"exploration_commit={committed_file}:{committed_line}, file_exists={file_exists}, valid_line={valid_line}, "
            f"evidence_steps={len(self._state.history)}, distinct_actions={distinct_actions}, reward={reward:.2f}"
        )
        breakdown = RewardBreakdown(
            reward=reward,
            base_step_penalty=base_step_penalty,
            premature_commit_penalty=premature_commit_penalty,
            file_score=file_score,
            line_score=line_score,
            efficiency_bonus_score=efficiency_bonus_score,
            correct_file=file_exists,
            correct_line=valid_line,
            efficiency_bonus=efficiency_bonus_score > 0,
            evidence_sufficient=evidence_sufficient,
            reason=reason,
        )
        return reason, breakdown

    def _obs(self) -> BugHuntObservation:
        return BugHuntObservation(
            file_tree=list(self._state.file_tree),
            stderr=self._state.stderr,
            failing_test=self._state.failing_test,
            last_tool_output=self._state.last_tool_output,
            steps_left=self._state.max_steps - self._state.steps_used,
            history=list(self._state.history),
        )

    def _info(self, reward_breakdown: Optional[RewardBreakdown] = None) -> Dict[str, Any]:
        return {
            "case_name": self.label,
            "steps_used": self._state.steps_used if self._state else 0,
            "max_steps": self.max_steps,
            "bug_file": None,
            "bug_line": None,
            "bug_summary": f"Extracted workspace ({self.source_type})",
            "reward_breakdown": reward_breakdown.as_dict() if reward_breakdown else None,
        }

    def _parse_action(self, action) -> Tuple[int, Dict[str, Any]]:
        if isinstance(action, tuple):
            action_id, params = action
            return int(action_id), dict(params)
        return int(action), {}

    def _format_history(self, action_id: int, params: Dict[str, Any], output: str, reward: float) -> str:
        names = {
            0: "read_file",
            1: "run_test",
            2: "search_symbol",
            3: "trace_caller",
            4: "commit_location",
        }
        short_output = output.replace("\n", " ")[:180]
        return f"{names.get(action_id, 'unknown')}({params}) -> reward={reward:.3f} -> {short_output}"

    def _require_reset(self):
        if self._state is None:
            raise RuntimeError("Call env.reset() before env.step().")
