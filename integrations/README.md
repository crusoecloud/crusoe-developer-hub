# Integrations

Connect Crusoe to the frameworks, editors, and services you already use. Start with the tool you want to connect, then follow its setup guide.

Integrations organize content around a third-party tool or platform. [Examples](../examples/README.md) organize standalone tasks around a developer outcome, such as provisioning infrastructure or serving a model. A demo that primarily teaches a framework connection belongs here, even when it includes a complete application.

## Choose an integration

The included content column describes what is checked into this repository. These entries contain source code, configuration, or setup instructions; they are not planned placeholders. Live model availability and compatibility still depend on your installed tools and account.

| <div align="center">Developer goal</div> | <div align="center">Integration</div> | <div align="center">Included content</div> | <div align="center">Prerequisites</div> |
|---|---|---|---|
| Call Crusoe from LangChain | [LangChain](langchain/README.md) | Python package, usage guide, unit and integration tests | Python 3.9+, Crusoe Inference API key; Poetry for package development |
| Route clients through a local gateway | [LiteLLM](litellm/README.md) | Docker Compose gateway, optional PostgreSQL overlay, client scripts | Docker with Compose, Crusoe Inference API key |
| Use MLflow's deployment interface | [MLflow](mlflow/README.md) | Python deployment plugin, example script, tests | Python 3.9+, Crusoe Inference API key |
| Build an ADK tool-calling agent | [Google ADK](google-adk/README.md) | Time-query agent using ADK and LiteLLM | Python, dependencies in the guide, Crusoe Inference API key |
| Explore interactive apps on Spaces | [Hugging Face](huggingface/README.md) | Gradio, Streamlit, and Docker demos with individual READMEs | Per-demo Python or Docker dependencies and credentials; Hugging Face account for deployment |
| Configure an editor | [Cursor](cursor/README.md) | Editor setup guide | Cursor, Crusoe Inference API key |
| Configure an editor | [Zed](zed/README.md) | Editor setup guide and settings JSON | Zed, Crusoe Inference API key |
| Ground answers in search results | [Linkup](linkup/README.md) | Complete Python recipe in the README | Python, Crusoe and Linkup API keys |
| Build type-safe agents with Pydantic AI | [Pydantic AI](pydantic-ai/README.md) | Basic, tool-use, and structured-output scripts on the native `crusoe:` provider | Python 3.10+, Crusoe Inference API key |
| Run Strands Agents on Crusoe | [Strands Agents](strands-agents/README.md) | Agent script that chains two custom tools | Python 3.10+, Crusoe Inference API key |
| Build Agno agents on Crusoe | [Agno](agno/README.md) | Basic, tool-use, and structured-output scripts via `OpenAILike` | Python 3.10+, Crusoe Inference API key |
| Swap providers with one string | [aisuite](aisuite/README.md) | Chat and automatic tool-loop scripts on the built-in `crusoe:` provider | Python 3.10+, Crusoe Inference API key |
| Call Crusoe from TypeScript | [Vercel AI SDK](vercel-ai-sdk/README.md) | Generate, stream, tool, and structured-output scripts with `@ai-sdk/openai-compatible` | Node.js 20+, Crusoe Inference API key |
| Reuse vLLM client code | [vLLM](vllm/README.md) | Streaming, guided decoding, and catalog scripts; a client that targets Crusoe or your own `vllm serve` | Python 3.10+, Crusoe Inference API key |
| Write SGLang programs | [SGLang](sglang/README.md) | Multi-turn, fork, batch, and streaming program on the built-in `Crusoe` backend | Python 3.10+, Crusoe Inference API key |
| Build a voice agent with Pipecat | [Pipecat](pipecat/README.md) | Function-calling voice bot on `CrusoeLLMService` | Python 3.10+, Crusoe, Deepgram, and Cartesia keys |
| Build a voice agent with LiveKit | [LiveKit Agents](livekit/README.md) | Voice agent with a function tool through the OpenAI plugin | Python 3.10+, Crusoe, LiveKit, Deepgram, and Cartesia keys |
| Code with an AI agent | [Cline](cline/README.md) | Setup guide for the built-in Crusoe provider | Cline, Crusoe Inference API key |
| Code with an AI agent | [Kilo Code](kilo-code/README.md) | Setup guide and `kilo.json` config | Kilo Code, Crusoe Inference API key |
| Code with an AI agent | [OpenCode](opencode/README.md) | Setup guide and `opencode.json` config | OpenCode, Crusoe Inference API key |
| Code with an AI agent | [Continue](continue/README.md) | `config.yaml` for the chat, edit, and apply roles | Continue, Crusoe Inference API key |
| Run Codex CLI or Claude Code on Crusoe | [opencodex](opencodex/README.md) | Setup guide for the built-in Crusoe preset | Node.js 20+, Crusoe Inference API key |
| Route through an AI gateway | [Portkey](portkey/README.md) | Gateway headers, SDK client, and a fallback config | Docker or a Portkey account, Crusoe Inference API key |
| Observe requests and costs | [Helicone](helicone/README.md) | Gateway client with tracing headers | Helicone account, Crusoe Inference API key |

## Start with a request

For API key setup and your first request, follow the official [Crusoe inference quickstart](https://docs.crusoecloud.com/serverless-inference/index.html). For a local gateway, follow the [LiteLLM quickstart](litellm/README.md#quickstart-60-seconds-no-database). To work inside an existing Python application, start with [LangChain](langchain/README.md#quick-start) or [Google ADK](google-adk/README.md#quick-start). For an agent framework, [Pydantic AI](pydantic-ai/README.md) and [Agno](agno/README.md) each ship three runnable recipes, and [Strands Agents](strands-agents/README.md) ships one. For TypeScript, start with the [Vercel AI SDK](vercel-ai-sdk/README.md).

Crusoe is also a native provider in several of these tools upstream. Where an upstream pull request is still in review, the guide says so and shows the configuration that works today.

Most integrations require a Crusoe account and an API key. Follow the chosen guide and the official [API key setup instructions](https://docs.crusoecloud.com/serverless-inference/index.html) to configure credentials. Python integrations commonly read:

```bash
export CRUSOE_API_KEY="your-api-key"
```

Editors have their own key settings. The [Hugging Face setup notes](huggingface/README.md#setup) identify demos with different key names or additional providers. Keep real credentials out of committed files.

Inference requests can incur usage charges. Search providers and hosted Spaces can add their own costs. The LiteLLM smoke script makes a live completion request. Read the selected guide before running or deploying a demo, and stop local services when finished.

[Back to the developer hub](../README.md)
