from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import random

import gymnasium as gym
from gymnasium import spaces

from bughunt_env.fixtures import BugCase, get_cases
from bughunt_env.reward import RewardBreakdown, budget_exhausted_reward, score_commit, tool_step_reward


@dataclass
class BugHuntState:
    case_name: str
    file_tree: List[str]
    stderr: str
    failing_test: str
    steps_used: int = 0
    max_steps: int = 10
    last_tool_output: str = ""
    history: List[str] = field(default_factory=list)
    done: bool = False


@dataclass
class BugHuntObservation:
    file_tree: List[str]
    stderr: str
    failing_test: str
    last_tool_output: str
    steps_left: int
    history: List[str]


class BugHuntEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps: int = 10, seed: Optional[int] = None):
        super().__init__()
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.cases: List[BugCase] = get_cases()
        self.case_index = {case.name: case for case in self.cases}
        self.case: Optional[BugCase] = None
        self._state: Optional[BugHuntState] = None

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({"steps_left": spaces.Discrete(max_steps + 1)})

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)

        case_name = (options or {}).get("case_name")
        self.case = self.case_index.get(case_name) if case_name else self.rng.choice(self.cases)
        if self.case is None:
            raise KeyError(f"Unknown case_name: {case_name}")

        shuffled_tree = list(self.case.file_tree)
        self.rng.shuffle(shuffled_tree)
        self._state = BugHuntState(
            case_name=self.case.name,
            file_tree=shuffled_tree,
            stderr=self.case.stderr,
            failing_test=self.case.failing_test,
            max_steps=self.max_steps,
        )
        return self._obs(), self._info()

    def state(self) -> BugHuntState:
        self._require_reset()
        return self._state

    def step(self, action):
        self._require_reset()
        if self._state.done:
            return self._obs(), 0.0, True, False, self._info()

        action_id, params = self._parse_action(action)
        self._state.steps_used += 1

        reward = tool_step_reward()
        terminated = False
        truncated = False
        reward_breakdown: Optional[RewardBreakdown] = None
        output = self._execute_action(action_id, params)

        if action_id == 4:
            committed_file = params.get("file", "")
            committed_line = int(params.get("line", -999))
            reward_breakdown = score_commit(
                committed_file=committed_file,
                committed_line=committed_line,
                target_file=self.case.bug_file,
                target_line=self.case.bug_line,
                steps_used=self._state.steps_used,
                max_steps=self.max_steps,
                prior_evidence_steps=len(self._state.history),
            )
            reward = reward_breakdown.reward
            output = reward_breakdown.reason
            terminated = True
            self._state.done = True
        elif self._state.steps_used >= self.max_steps:
            reward = budget_exhausted_reward(self._state.steps_used)
            output += "\nBudget exhausted. Episode failed."
            truncated = True
            self._state.done = True

        self._state.last_tool_output = output
        self._state.history.append(self._format_history(action_id, params, output, reward))
        return self._obs(), reward, terminated, truncated, self._info(reward_breakdown=reward_breakdown)

    def render(self):
        self._require_reset()
        print("=== BugHuntEnv ===")
        print(f"Case: {self._state.case_name}")
        print(f"Failing test: {self._state.failing_test}")
        print(f"Steps: {self._state.steps_used}/{self._state.max_steps}")
        print(f"Last output:\n{self._state.last_tool_output}")

    def _execute_action(self, action_id: int, params: Dict[str, Any]) -> str:
        if action_id == 0:
            return self._read_file(params.get("path", ""))
        if action_id == 1:
            return self._run_test(params.get("name", self.case.failing_test))
        if action_id == 2:
            return self._search_symbol(params.get("name", ""))
        if action_id == 3:
            return self._trace_caller(params.get("fn", ""))
        if action_id == 4:
            return "Committing bug location..."
        return "Invalid action."

    def _read_file(self, path: str) -> str:
        if path not in self.case.files:
            return f"File not found: {path}"
        return f"--- {path} ---\n{self.case.files[path]}"

    def _run_test(self, name: str) -> str:
        if name == self.case.failing_test or name == "all":
            return self.case.stderr
        return f"No failing output found for test: {name}"

    def _search_symbol(self, name: str) -> str:
        if name in self.case.symbols:
            return f"Symbol `{name}` found in {self.case.symbols[name]}"
        return f"Symbol `{name}` not found."

    def _trace_caller(self, fn: str) -> str:
        if fn in self.case.callers:
            return self.case.callers[fn]
        return f"No caller trace found for `{fn}`."

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
            "case_name": self.case.name if self.case else None,
            "steps_used": self._state.steps_used if self._state else 0,
            "max_steps": self.max_steps,
            "bug_file": self.case.bug_file if self.case else None,
            "bug_line": self.case.bug_line if self.case else None,
            "bug_summary": self.case.bug_summary if self.case else None,
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
        short_output = output.replace("\n", " ")[:160]
        return f"{names.get(action_id, 'unknown')}({params}) -> reward={reward} -> {short_output}"

    def _require_reset(self):
        if self._state is None or self.case is None:
            raise RuntimeError("Call env.reset() before env.step().")
