from dataclasses import dataclass


@dataclass
class RewardBreakdown:
    reward: float
    correct_file: bool
    correct_line: bool
    efficiency_bonus: bool
    reason: str


def score_commit(
    committed_file: str,
    committed_line: int,
    target_file: str,
    target_line: int,
    steps_used: int,
    max_steps: int,
) -> RewardBreakdown:
    reward = 0.0
    correct_file = committed_file == target_file
    correct_line = abs(committed_line - target_line) <= 5
    efficiency_bonus = steps_used <= 3

    reward -= 0.1 * steps_used

    if correct_file:
        reward += 1.0
    else:
        reward -= 1.0

    if correct_file and correct_line:
        reward += 0.5

    if correct_file and correct_line and efficiency_bonus:
        reward += 0.4

    reason = (
        f"commit={committed_file}:{committed_line}, "
        f"target={target_file}:{target_line}, "
        f"correct_file={correct_file}, correct_line={correct_line}, "
        f"steps={steps_used}/{max_steps}, reward={reward:.2f}"
    )

    return RewardBreakdown(
        reward=round(reward, 3),
        correct_file=correct_file,
        correct_line=correct_line,
        efficiency_bonus=efficiency_bonus and correct_file and correct_line,
        reason=reason,
    )


def budget_exhausted_reward(steps_used: int) -> float:
    return round(-0.5 - (0.1 * steps_used), 3)


def tool_step_reward() -> float:
    return -0.1
