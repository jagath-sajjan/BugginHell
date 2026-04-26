from __future__ import annotations

import random
import re


class RandomAgent:
    def act(self, obs):
        action_id = random.randint(0, 4)
        if action_id == 0:
            return (0, {"path": random.choice(obs.file_tree)})
        if action_id == 1:
            return (1, {"name": obs.failing_test})
        if action_id == 2:
            return (2, {"name": obs.failing_test})
        if action_id == 3:
            return (3, {"fn": random.choice(["calculate_total", "is_admin", "normalize_name"])})
        return (4, {"file": random.choice(obs.file_tree), "line": random.randint(1, 20)})


class StrategicBugHunter:
    def __init__(self):
        self.phase = 0
        self.target_symbol = None
        self.target_file = None

    def act(self, obs):
        if self.phase == 0:
            self.phase += 1
            return (2, {"name": obs.failing_test})
        if self.phase == 1:
            self.target_symbol = self._extract_symbol_from_error(obs.stderr)
            self.phase += 1
            return (3, {"fn": self.target_symbol})
        if self.phase == 2:
            self.target_file = self._extract_file_from_output(obs.last_tool_output, obs.file_tree)
            self.phase += 1
            return (0, {"path": self.target_file})
        bug_line = self._guess_bug_line(obs.last_tool_output)
        return (4, {"file": self.target_file or obs.file_tree[0], "line": bug_line})

    def _extract_symbol_from_error(self, stderr):
        matches = re.findall(r"= ([a-zA-Z_][a-zA-Z0-9_]*)\(", stderr)
        if matches:
            return matches[-1]
        for item in ["calculate_total", "is_admin", "normalize_name", "safe_divide"]:
            if item in stderr:
                return item
        return "calculate_total"

    def _extract_file_from_output(self, output, file_tree):
        for file in file_tree:
            if file in output and file.endswith(".py") and not file.startswith("tests/"):
                return file
        for file in file_tree:
            if file.endswith(".py") and not file.startswith("tests/"):
                return file
        return file_tree[0]

    def _guess_bug_line(self, file_output):
        bug_keywords = ["range(len", "!=", "==", "- 1", "+ 1", "return True", "return False", "None"]
        for idx, line in enumerate(file_output.splitlines(), start=1):
            if any(keyword in line for keyword in bug_keywords):
                return max(1, idx - 1)
        return 1
