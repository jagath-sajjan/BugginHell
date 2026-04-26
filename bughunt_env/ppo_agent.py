from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


ACTION_DIM = 5
OBS_DIM = 13
FEATURE_NAMES = [
    "steps_left",
    "failed",
    "assert",
    "zerodivision",
    "error",
    "none",
    "bool_logic",
    "has_src_file",
    "explored",
    "explored_deep",
    "symbol_hit",
    "symbol_miss",
    "caller_hit",
]
ACTION_NAMES = [
    "read_file",
    "run_test",
    "search_symbol",
    "trace_caller",
    "commit_location",
]


@dataclass
class PolicyStep:
    action_id: int
    action: Tuple[int, Dict[str, Any]]
    log_prob: torch.Tensor
    value: torch.Tensor
    probs: torch.Tensor
    entropy: torch.Tensor
    features: torch.Tensor
    h1: torch.Tensor
    h2: torch.Tensor
    logits: torch.Tensor
    c1: torch.Tensor
    c2: torch.Tensor
    c3: torch.Tensor

    def snapshot(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_name": ACTION_NAMES[self.action_id],
            "features": self.features.tolist(),
            "feature_names": FEATURE_NAMES,
            "h1": self.h1.tolist(),
            "h2": self.h2.tolist(),
            "logits": self.logits.tolist(),
            "probs": self.probs.tolist(),
            "value": float(self.value.item()),
            "entropy": float(self.entropy.item()),
            "c1": self.c1.tolist(),
            "c2": self.c2.tolist(),
            "c3": self.c3.tolist(),
        }


def encode_obs(obs):
    """
    Encode observation into a fixed-size feature vector.
    Features are generic signals — keyword presence, step budget —
    NOT hardcoded case-specific names.
    """
    stderr_lower = obs.stderr.lower()
    tool_lower = obs.last_tool_output.lower()
    combined = " ".join([
        " ".join(obs.file_tree),
        obs.failing_test,
        stderr_lower,
        tool_lower,
        str(obs.steps_left),
    ])

    return torch.tensor([
        obs.steps_left / 10,                                         # budget remaining
        float("failed" in combined),                                 # test is failing
        float("assert" in combined),                                 # assertion error
        float("zerodivision" in combined),                           # divide by zero
        float("error" in combined),                                  # generic error
        float("none" in combined),                                   # None-related
        float("true" in combined or "false" in combined),            # bool logic
        float(any(f.endswith(".py") and "test" not in f for f in obs.file_tree)),
        float(len(obs.history) > 0),                                 # has explored
        float(len(obs.history) > 2),                                 # explored a lot
        float("found in" in tool_lower),                             # symbol search hit
        float("not found" in tool_lower),                            # symbol search miss
        float("calls" in tool_lower),                                # trace_caller hit
    ], dtype=torch.float32)


def build_action(action_id: int, obs):
    """
    Convert a policy action_id into an actual env action tuple.
    Params are derived from the current observation — not hardcoded.
    """
    stderr = obs.stderr.lower()
    tool_out = obs.last_tool_output.lower()

    # Derive best symbol guess from stderr / last tool output
    symbol_matches = re.findall(r'\b([a-z_][a-z0-9_]{3,})\s*[\(=]', stderr + " " + tool_out)
    noise = {"assert", "where", "from", "import", "true", "false", "none", "with", "raise"}
    candidates = [s for s in symbol_matches if s not in noise]
    guessed_symbol = candidates[0] if candidates else obs.failing_test

    # Derive best file guess from tool output patterns
    file_match = re.search(r'found in ([^\s]+\.py)', tool_out)
    caller_match = re.search(r'from ([^\s]+\.py)', tool_out)
    src_files = [f for f in obs.file_tree if f.endswith(".py") and "test" not in f]

    if file_match:
        guessed_file = file_match.group(1)
    elif caller_match:
        guessed_file = caller_match.group(1)
    elif src_files:
        guessed_file = src_files[0]
    else:
        guessed_file = obs.file_tree[0]

    # Derive line guess from tool output
    line_match = re.search(r'line[:\s]+(\d+)', tool_out)
    if line_match:
        guessed_line = int(line_match.group(1))
    else:
        bug_keywords = ["range(len", "!=", "==", "- 1", "+ 1", "return True", "return False", "None"]
        guessed_line = 1
        for idx, line in enumerate(obs.last_tool_output.splitlines(), start=1):
            if any(kw in line for kw in bug_keywords):
                guessed_line = max(1, idx - 1)
                break

    if action_id == 0:
        return (0, {"path": guessed_file})
    if action_id == 1:
        return (1, {"name": obs.failing_test})
    if action_id == 2:
        return (2, {"name": guessed_symbol})
    if action_id == 3:
        return (3, {"fn": guessed_symbol})
    return (4, {"file": guessed_file, "line": guessed_line})


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(OBS_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, ACTION_DIM)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        h1 = self.shared[0:2](x)
        h2 = self.shared[2:4](h1)
        actor_logits = self.actor(h2)
        value = self.critic(h2)
        return actor_logits, value.squeeze(-1), h1, h2

    def contribution_matrices(self, x, h1, h2):
        layer1 = self.shared[0]
        layer2 = self.shared[2]
        c1 = x.unsqueeze(1) * layer1.weight.detach().transpose(0, 1)
        c2 = h1.unsqueeze(1) * layer2.weight.detach().transpose(0, 1)
        c3 = h2.unsqueeze(1) * self.actor.weight.detach().transpose(0, 1)
        return c1, c2, c3

    def act(self, obs):
        x = encode_obs(obs)
        logits, value, h1, h2 = self.forward(x)
        c1, c2, c3 = self.contribution_matrices(x, h1, h2)
        dist = Categorical(logits=logits)
        action_id = dist.sample()
        return PolicyStep(
            action_id=action_id.item(),
            action=build_action(action_id.item(), obs),
            log_prob=dist.log_prob(action_id),
            value=value,
            probs=dist.probs.detach(),
            entropy=dist.entropy().detach(),
            features=x.detach(),
            h1=h1.detach(),
            h2=h2.detach(),
            logits=logits.detach(),
            c1=c1.detach(),
            c2=c2.detach(),
            c3=c3.detach(),
        )

    def inspect(self, obs):
        with torch.no_grad():
            x = encode_obs(obs)
            logits, value, h1, h2 = self.forward(x)
            c1, c2, c3 = self.contribution_matrices(x, h1, h2)
            probs = torch.softmax(logits, dim=-1)
        return {
            "features": x.tolist(),
            "feature_names": FEATURE_NAMES,
            "logits": logits.tolist(),
            "probs": probs.tolist(),
            "value": float(value.item()),
            "h1": h1.tolist(),
            "h2": h2.tolist(),
            "c1": c1.tolist(),
            "c2": c2.tolist(),
            "c3": c3.tolist(),
            "suggested_action_id": int(torch.argmax(probs).item()),
        }


class PPOTrainer:
    def __init__(self, env_cls, lr=3e-4, gamma=0.99, clip_eps=0.2):
        self.env_cls = env_cls
        self.model = ActorCritic()
        self.optim = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps

    def collect_episode(self, seed=None):
        env = self.env_cls(seed=seed)
        obs, _ = env.reset(seed=seed)

        states, actions, old_logps, rewards, values = [], [], [], [], []
        probs_log, entropies, env_actions, case_names, activations_log = [], [], [], [], []

        for _ in range(env.max_steps):
            step = self.model.act(obs)
            x = step.features
            action_id = step.action_id
            env_action = step.action

            next_obs, reward, term, trunc, _ = env.step(env_action)

            states.append(x)
            actions.append(action_id)
            old_logps.append(step.log_prob.detach())
            rewards.append(float(reward))
            values.append(step.value.detach())
            probs_log.append(step.probs.tolist())
            entropies.append(float(step.entropy.item()))
            env_actions.append(env_action)
            case_names.append(env.case.name)
            activations_log.append(step.snapshot())

            obs = next_obs

            if term or trunc:
                break

        return states, actions, old_logps, rewards, values, probs_log, env_actions, entropies, case_names, activations_log

    def update(self, batch, epochs=4):
        states, actions, old_logps, rewards, values, _, _, _, _, _ = batch

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        states = torch.stack(states)
        actions = torch.tensor(actions)
        old_logps = torch.stack(old_logps)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values)

        advantages = returns - values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        final_loss = 0.0

        for _ in range(epochs):
            logits, new_values, _, _ = self.model(states)
            dist = Categorical(logits=logits)
            new_logps = dist.log_prob(actions)

            ratio = torch.exp(new_logps - old_logps)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            actor_loss = -torch.min(unclipped, clipped).mean()
            critic_loss = (returns - new_values).pow(2).mean()
            entropy = dist.entropy().mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            final_loss = loss.item()

        return final_loss

    def train(self, episodes=100):
        logs = []

        for ep in range(episodes):
            batch = self.collect_episode(seed=ep)
            loss = self.update(batch)
            total_reward = sum(batch[3])
            success = any(action[0] == 4 for action in batch[6]) and total_reward > 0

            logs.append({
                "episode": ep,
                "reward": total_reward,
                "loss": loss,
                "actions": batch[6],
                "probs": batch[5],
                "entropy": sum(batch[7]) / max(1, len(batch[7])),
                "success": success,
                "case_name": batch[8][0] if batch[8] else None,
                "activations": batch[9],
            })

        return logs
