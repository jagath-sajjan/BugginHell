from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv(Path(".env"))

hf_token = os.getenv("HF_TOKEN")
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

from bughunt_env import BugHuntEnv, LocalLLMAgent

env = BugHuntEnv(seed=1)
obs, info = env.reset(seed=1)

agent = LocalLLMAgent(model_name=model_name)

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
