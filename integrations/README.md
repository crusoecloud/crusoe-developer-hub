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

## Start with a request

For API key setup and your first request, follow the official [Crusoe inference quickstart](https://docs.crusoecloud.com/serverless-inference/index.html). For a local gateway, follow the [LiteLLM quickstart](litellm/README.md#quickstart-60-seconds-no-database). To work inside an existing Python application, start with [LangChain](langchain/README.md#quick-start) or [Google ADK](google-adk/README.md#quick-start).

Most integrations require a Crusoe account and an API key. Follow the chosen guide and the official [API key setup instructions](https://docs.crusoecloud.com/serverless-inference/index.html) to configure credentials. Python integrations commonly read:

```bash
export CRUSOE_API_KEY="your-api-key"
```

Editors have their own key settings. The [Hugging Face setup notes](huggingface/README.md#setup) identify demos with different key names or additional providers. Keep real credentials out of committed files.

Inference requests can incur usage charges. Search providers and hosted Spaces can add their own costs. The LiteLLM smoke script makes a live completion request. Read the selected guide before running or deploying a demo, and stop local services when finished.

[Back to the developer hub](../README.md)
