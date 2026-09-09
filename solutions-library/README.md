# Solutions Library

Workflows that combine several Crusoe components into something you can deploy: GPU clusters with their networking checks, shared storage, Slurm images, model serving, distributed pretraining, and monitoring. Each solution is a self-contained directory with its own prerequisites, configuration, and cleanup notes, maintained by Crusoe Solutions Engineering.

Browse the full library at [github.com/crusoecloud/solutions-library](https://github.com/crusoecloud/solutions-library). A few starting points:

| <div align="center">You want to</div> | <div align="center">Start with</div> |
| --- | --- |
| Bring up GPU VMs and confirm multi-node communication | [Create VMs and run an NCCL test](https://github.com/crusoecloud/solutions-library/tree/main/create-vms-and-run-nccl-test) |
| Mount shared storage across a cluster | [Shared volumes driver setup](https://github.com/crusoecloud/solutions-library/tree/main/shared-volumes-driver-setup) |
| Build a custom image with Slurm binaries | [Slurm custom image](https://github.com/crusoecloud/solutions-library/tree/main/slurm-custom-image) |
| Serve a model on Kubernetes | [KServe example](https://github.com/crusoecloud/solutions-library/tree/main/crusoe-kserve-example) |
| Pretrain Llama 3.1 with TorchTitan on Kubernetes | [TorchTitan PyTorchJob](https://github.com/crusoecloud/solutions-library/tree/main/torchtitan-llama3_1-kubernetes-pytorchjob) |
| Monitor cluster resources | [Grafana on Crusoe Managed Kubernetes](https://github.com/crusoecloud/solutions-library/tree/main/grafana-cmk) |

Read a solution's README before applying it. Most create paid compute, storage, or networking resources, and cleanup is specific to each one. For the concepts behind these workflows, see [Foundations](../foundations/); for single-task versions, see [Examples](../examples/).

[Back to the hub](../README.md)
