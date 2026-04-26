import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

model_name = os.getenv("MODEL_NAME", "qwen/qwen3-32b")

from bughunt_env import BugHuntEnv, LocalLLMAgent

env = BugHuntEnv(seed=1)
obs, info = env.reset(seed=1)

agent = LocalLLMAgent(
    model_name=model_name,
    api_key=os.getenv("HACKCLUB_API_KEY") or os.getenv("OPENAI_API_KEY"),
)

print("CASE:", info["case_name"])
print("FAILING:", obs.failing_test)

total = 0

for step in range(env.max_steps):
    action, raw = agent.act(obs)

    print("\nRAW LLM:")
    print(raw)
    print("PARSED ACTION:", action)

    obs, reward, terminated, truncated, _ = env.step(action)
    total += reward

    print("REWARD:", reward)
    print("OUTPUT:")
    print(obs.last_tool_output)

    if terminated or truncated:
        break

print("\nTOTAL:", round(total, 3))
