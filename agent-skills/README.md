# Crusoe Agent Skills

Agent skills give AI coding agents such as Claude Code, Cursor, Codex and GitHub Copilot
working knowledge of Crusoe Cloud. A skill is a folder with a `SKILL.md` at its root: the YAML
frontmatter tells the agent when to load it, the body tells it what to do. The rest of the
folder is read only when a task calls for it, so a large skill costs nothing until it is used.

[Documentation](https://docs.crusoe.ai/) &nbsp;|&nbsp;
[Managed AI](https://docs.crusoe.ai/managed-ai/overview/) &nbsp;|&nbsp;
[API reference](https://docs.crusoe.ai/api/managed-ai/) &nbsp;|&nbsp;
[Console](https://console.crusoecloud.com/)

## Available

| Group | Skill | What it does |
|---|---|---|
| Managed AI | [`crusoe-training`](crusoe-training/) | Prepare and validate chat datasets, launch Serverless Fine-Tuning jobs, pick LoRA checkpoints, deploy them to Self-Serve Deployments, and run inference against the result |

## Install

Copy the folder into wherever your agent looks for skills.

```bash
# Claude Code, available in every project
cp -R agent-skills/crusoe-training ~/.claude/skills/

# Claude Code, this project only
cp -R agent-skills/crusoe-training .claude/skills/
```

For other agents (Cursor, Codex, GitHub Copilot, pi), copy the folder into that agent's
skills directory, for example `~/.agents/skills/`, or point the agent at
`crusoe-training/SKILL.md`.

To check that it loaded, ask for something it should recognise, such as "fine-tune Qwen on my
dataset with Crusoe".

## Adding a skill

One folder per skill, with `SKILL.md` at its root:

```text
agent-skills/
├── README.md
└── <skill-name>/
    ├── SKILL.md        # agent instructions
    ├── scripts/        # optional: helper scripts the agent runs
    └── references/     # optional: docs the agent reads on demand
```

A skill that covers several distinct workflows may nest sub-skills in its own `skills/`
folder. The top-level `SKILL.md` routes to them, so the skill is still installed and
discovered as one folder. `crusoe-training` is the worked example.

The frontmatter `description` is the only part of a skill an agent sees before deciding
whether to load it, so write it as a trigger rather than a summary: say what the skill does,
then name the situations and phrasings that should pull it in.

```markdown
---
name: crusoe-example-skill
description: One sentence on what this skill does, then the situations and phrasings that should trigger it.
---

# Crusoe example skill

Step-by-step instructions for the agent.
```

Keep `SKILL.md` short enough to stay loaded. Detail that is needed only sometimes belongs in
`references/`, and anything deterministic belongs in `scripts/`, where the agent runs it
instead of reasoning it out.
