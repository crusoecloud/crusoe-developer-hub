# Examples

Build toward a concrete outcome. Examples are organized by developer task, with infrastructure, training, and inference as the primary paths. Each implementation has one home and its own prerequisites, setup, expected output, and cleanup.

| <div align="center">Goal</div> | <div align="center">Section</div> | <div align="center">Available now</div> |
| --- | --- | --- |
| Provision compute, prepare storage, or operate clusters | [Infrastructure](infrastructure/) | [GB200 NVL72 validation with Terraform and Ansible](infrastructure/gb200-nvl72-validation/); links to related solutions |
| Prepare data, train, fine-tune, or evaluate models | [Training](training/) | [PII redaction notebook](training/pii-redaction/) and [multipart dataset upload notebook](training/multipart-dataset-upload/) |
| Call models or build a serving workflow | [Inference](inference/) | [MLPerf Inference v6.1 on AMD MI355X](inference/mlperf-inference-amd-mi355x-v6.1/) and [AI Infrastructure SimCity](inference/ai-infrastructure-simcity/); related integrations and serving solutions linked |
| Follow an end-to-end event session | [Workshop](workshop/) | [Serve, Measure, Tune: SGLang on Crusoe Managed Kubernetes](workshop/sglang-kubernetes-workshop/): deploy SGLang on Kubernetes with KServe, benchmark it under realistic load, and tune KV cache precision and scheduling on your own GPU. 90 min session (about 60 min hands-on) |
| Apply a focused recipe | [Cookbooks](cookbooks/) | [Managed MLflow](cookbooks/managed-mlflow/) |
| Give a coding assistant a skill that runs on Crusoe models | [Skills](skills/) | [eBay Fashion Tracker](skills/ebay-fashion-tracker/) |
| Build at a hackathon, or see what teams built on Crusoe | [Hackathons](hackathons/) | [Recursive Agents Hackathon kit](hackathons/crusoe-cv-recursive-agents-hackathon/), plus the projects from [The AI Conference Hack Day 2026](hackathons/the-ai-conference-hack-day-2026/) and the [Recursive Agents Hackathon](hackathons/crusoe-cv-recursive-agents-hackathon/projects.md) |

Start with the selected entry's README, then run its commands from the specified directory. Dependencies, credentials, hardware, model access, and costs vary; there is no common installation for all examples.

[Foundations](../foundations/) explain the concepts behind these workflows. [Integrations](../integrations/) focus on connecting a named tool to Crusoe. Broader infrastructure and operations workflows live in the [Solutions Library](https://github.com/crusoecloud/solutions-library). Link to related content instead of duplicating it.

[Back to the hub](../README.md)
