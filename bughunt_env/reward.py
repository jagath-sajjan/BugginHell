from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    base_step_penalty: float
    premature_commit_penalty: float
    file_score: float
    line_score: float
    efficiency_bonus_score: float
    correct_file: bool
    correct_line: bool
    efficiency_bonus: bool
    evidence_sufficient: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "reward": self.reward,
            "base_step_penalty": self.base_step_penalty,
            "premature_commit_penalty": self.premature_commit_penalty,
            "file_score": self.file_score,
            "line_score": self.line_score,
            "efficiency_bonus_score": self.efficiency_bonus_score,
            "correct_file": self.correct_file,
            "correct_line": self.correct_line,
            "efficiency_bonus": self.efficiency_bonus,
            "evidence_sufficient": self.evidence_sufficient,
            "reason": self.reason,
        }


def score_commit(
    committed_file: str,
    committed_line: int,
    target_file: str,
    target_line: int,
    steps_used: int,
    max_steps: int,
    prior_evidence_steps: int,
) -> RewardBreakdown:
    evidence_sufficient = prior_evidence_steps >= 2
    correct_file = committed_file == target_file
    correct_line = committed_line == target_line
    efficiency_bonus = evidence_sufficient and 2 <= steps_used <= 5 and correct_file and correct_line

    base_step_penalty = round(-0.1 * steps_used, 3)
    premature_commit_penalty = 0.0 if evidence_sufficient else -2.0
    file_score = 1.0 if correct_file else -1.0
    line_score = 0.75 if correct_file and correct_line else 0.0
    efficiency_bonus_score = 0.4 if efficiency_bonus else 0.0
    reward = round(
        base_step_penalty + premature_commit_penalty + file_score + line_score + efficiency_bonus_score,
        3,
    )

    reason = (
        f"commit={committed_file}:{committed_line}, "
        f"target={target_file}:{target_line}, "
        f"correct_file={correct_file}, correct_line={correct_line}, "
        f"evidence_steps={prior_evidence_steps}, steps={steps_used}/{max_steps}, reward={reward:.2f}"
    )

    return RewardBreakdown(
        reward=reward,
        base_step_penalty=base_step_penalty,
        premature_commit_penalty=premature_commit_penalty,
        file_score=file_score,
        line_score=line_score,
        efficiency_bonus_score=efficiency_bonus_score,
        correct_file=correct_file,
        correct_line=correct_line,
        efficiency_bonus=efficiency_bonus,
        evidence_sufficient=evidence_sufficient,
        reason=reason,
    )


def budget_exhausted_reward(steps_used: int) -> float:
    return round(-0.5 - (0.1 * steps_used), 3)


def tool_step_reward() -> float:
    return -0.1
