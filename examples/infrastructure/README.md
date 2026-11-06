# Infrastructure examples

Provision compute, prepare storage, configure networking, and operate clusters for developer workloads. Tasks include checking multi-node communication and mounting shared storage. Kubernetes, VMs, and Slurm can be implementation choices within a task.

## Examples in this section

| <div align="center">Task</div> | <div align="center">Example</div> | <div align="center">Prerequisites</div> |
| --- | --- | --- |
| Provision a GB200 NVL72 rack with Terraform, then validate IMEX, GEMM, and multi-node NVLink NCCL with Ansible | [GB200 NVL72 validation](gb200-nvl72-validation/) | GB200 capacity in one NVLink domain, Crusoe CLI, Terraform, Ansible |

## Related workflows in the Solutions Library

| <div align="center">Task</div> | <div align="center">Existing implementation in the Solutions Library</div> |
| --- | --- |
| Create GPU VMs and check collective communication | [Create VMs and run an NCCL test](https://github.com/crusoecloud/solutions-library/tree/main/create-vms-and-run-nccl-test) |
| Prepare VMs to mount shared storage | [Shared volumes driver setup](https://github.com/crusoecloud/solutions-library/tree/main/shared-volumes-driver-setup) |
| Build a custom image with Slurm binaries | [Slurm custom image](https://github.com/crusoecloud/solutions-library/tree/main/slurm-custom-image) |
| Monitor cluster resources | [Grafana deployment](https://github.com/crusoecloud/solutions-library/tree/main/grafana-cmk) |

Read each solution's prerequisites and inspect its configuration before applying it. These workflows can create or use paid compute, storage, and other resources; cleanup is specific to the chosen solution.

For concepts, see the planned [cloud foundations track](../../foundations/cloud/). Training jobs belong in [training](../training/), and model-serving workflows in [inference](../inference/), even when they use the same cluster tools.

[All examples](../README.md)
