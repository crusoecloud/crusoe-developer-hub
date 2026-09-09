# Using deep-repo-agent through the gateway

[deep-repo-agent](https://github.com/acheamponge/deep-repo-agent) is a CLI agent
that chats with any GitHub repository. It talks to Crusoe through the
`CRUSOE_API_BASE` environment variable — which means it can talk to this
gateway instead, with zero code changes:

```bash
pip install git+https://github.com/acheamponge/deep-repo-agent

export CRUSOE_API_BASE="http://localhost:4000/v1"
export CRUSOE_API_KEY="sk-local-dev-master"   # or a virtual key you minted

deep-repo-agent https://github.com/pallets/flask --model glm-5.2
```

Every tool-calling round-trip the agent makes now flows through the gateway:
budgeted by the virtual key, tracked in the spend logs, and the real Crusoe
API key never leaves the gateway host.
