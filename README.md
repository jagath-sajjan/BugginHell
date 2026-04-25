---
title: BugginHell
emoji: 🐛
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: server/app.py
pinned: false
---

# BugginHell, BugHunt RL

A reinforcement learning environment where agents learn to strategically localize hidden bugs in Python codebases.

## Core Idea

Most LLM coding agents either read everything or guess randomly.  
BugginHell teaches an agent to debug like a senior engineer:

1. inspect failing test
2. search relevant symbol
3. trace caller
4. read the correct file
5. commit the bug location

## Hackathon Stack

- PyTorch
- Gymnasium / OpenEnv style environment
- HuggingFace Spaces
- Gradio
- TRL GRPO training notebook
- GitHub-first reproducibility

## Demo Story

The app compares:

- **Untrained Base Agent**: wastes actions and commits the wrong file
- **Strategic RL Agent**: uses tool calls and localizes the bug quickly

> We used AI tooling to build an environment that teaches AI agents to debug better.
