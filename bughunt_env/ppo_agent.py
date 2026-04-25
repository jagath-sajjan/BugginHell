import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


ACTION_DIM = 5
OBS_DIM = 13


def encode_obs(obs):
    text = " ".join([
        " ".join(obs.file_tree),
        obs.failing_test,
        obs.stderr,
        obs.last_tool_output,
        str(obs.steps_left),
    ]).lower()

    return torch.tensor([
        obs.steps_left / 10,
        float("failed" in text),
        float("assert" in text),
        float("zerodivision" in text),
        float("calculate_total" in text),
        float("is_admin" in text),
        float("safe_divide" in text),
        float("normalize_name" in text),
        float("cart.py" in text),
        float("auth.py" in text),
        float("math_tools.py" in text),
        float("user_format.py" in text),
        float("tests/" in text),
    ], dtype=torch.float32)


def build_action(action_id, obs):
    text = (obs.stderr + " " + obs.last_tool_output).lower()

    if "calculate_total" in text:
        symbol, file, line = "calculate_total", "cart.py", 3
    elif "is_admin" in text:
        symbol, file, line = "is_admin", "auth.py", 2
    elif "safe_divide" in text:
        symbol, file, line = "safe_divide", "math_tools.py", 2
    elif "normalize_name" in text:
        symbol, file, line = "normalize_name", "user_format.py", 4
    else:
        symbol = obs.failing_test
        file = random.choice(obs.file_tree)
        line = random.randint(1, 8)

    if action_id == 0:
        return (0, {"path": file})
    if action_id == 1:
        return (1, {"name": obs.failing_test})
    if action_id == 2:
        return (2, {"name": symbol})
    if action_id == 3:
        return (3, {"fn": symbol})
    return (4, {"file": file, "line": line})


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
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, obs):
        x = encode_obs(obs)
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action_id = dist.sample()
        return action_id.item(), dist.log_prob(action_id), value, dist.probs.detach()


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

        states, actions, old_logps, rewards, values, probs_log, env_actions = [], [], [], [], [], [], []

        for _ in range(env.max_steps):
            x = encode_obs(obs)
            action_id, logp, value, probs = self.model.act(obs)
            env_action = build_action(action_id, obs)

            next_obs, reward, term, trunc, _ = env.step(env_action)

            states.append(x)
            actions.append(action_id)
            old_logps.append(logp.detach())
            rewards.append(float(reward))
            values.append(value.detach())
            probs_log.append(probs.tolist())
            env_actions.append(env_action)

            obs = next_obs

            if term or trunc:
                break

        return states, actions, old_logps, rewards, values, probs_log, env_actions

    def update(self, batch, epochs=4):
        states, actions, old_logps, rewards, values, _, _ = batch

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
            logits, new_values = self.model(states)
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

            logs.append({
                "episode": ep,
                "reward": total_reward,
                "loss": loss,
                "actions": batch[6],
                "probs": batch[5],
            })

        return logs
