# AI Engineering

This track explains how to build and reason about applications that use large language models. Topics covered or planned:

- Prompting, structured output, and tool calling as building blocks.
- Retrieval-augmented generation (RAG): connecting retrieval to generation and evaluating the resulting answers.
- Agents and agent loops: selecting actions, invoking tools, observing results, tracking state, and deciding when to stop.
- Graph-based workflows: making steps, branches, dependencies, and state transitions explicit.
- Agent harnesses: the surrounding runtime that manages tools, context, state, execution limits, and evaluation.
- Evaluation: checking retrieval quality, task completion, and failure cases as an application changes.

## Courses

| course | what you build |
|---|---|
| [Agents at Scale (with Open Models)](Courses/agents-at-scale/) | A research agent on open-weight models through Crusoe Managed Inference, from a hand-written tool-calling loop to MCP tools, a typed production agent, evals, a multi-agent team, a harness and a scheduled loop. Seven chapters, each folder runnable on its own. |

Lessons use small runnable exercises to teach a concept. Workflows that deliver a specific application or deployment outcome belong in [examples](../../examples/) and should link back here for background. Tool or framework configuration that connects another project to Crusoe belongs in [integrations](../../integrations/).

[All foundations](../README.md)
