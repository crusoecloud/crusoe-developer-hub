# GB200 NVL72 validation - one-command bring-up, test and teardown.
# Run from this directory. Terraform reads terraform/terraform.tfvars.

TF       := terraform -chdir=terraform
PLAYBOOK := ulimit -n 4096 && ansible-playbook
PYTHON   ?= python3

# AUTO_APPROVE=1 skips Terraform's confirmation prompts (make all AUTO_APPROVE=1).
ifeq ($(AUTO_APPROVE),1)
TF_APPROVE := -auto-approve
endif

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

.PHONY: all
all: up validate ## Provision the rack, then run prep, smoke, multinode and the report

.PHONY: deps
deps: ## Install the Ansible collection and Terraform providers
	ansible-galaxy collection install -r requirements.yml
	$(TF) init

terraform/terraform.tfvars:
	@echo "terraform/terraform.tfvars not found: cp terraform/terraform.tfvars.example terraform/terraform.tfvars and set project_id, ssh_public_key_path, nvlink_domain_id" >&2
	@exit 1

.PHONY: canary
canary: terraform/terraform.tfvars deps ## Create one GB200 VM to confirm image, capacity and placement
	$(TF) apply -var gpu_node_count=1 $(TF_APPROVE)

.PHONY: up
up: terraform/terraform.tfvars deps ## Create all GB200 VMs (gpu_node_count in terraform.tfvars)
	$(TF) apply -parallelism=20 $(TF_APPROVE)

# Smoke and multinode failures do not stop the run: multinode tests the nodes that passed
# smoke, and the report shows what failed.
.PHONY: validate
validate: ## Run prep, smoke and multinode, then write the report
	$(MAKE) prep
	-$(MAKE) smoke
	-$(MAKE) multinode
	$(MAKE) report

.PHONY: prep
prep: ## Wait for boot, set up node-to-node SSH and the IMEX domain
	$(PLAYBOOK) ansible/00-prep.yml

.PHONY: smoke
smoke: ## Per-node checks: environment, GEMM, single-node NCCL
	$(PLAYBOOK) ansible/10-smoke.yml

.PHONY: multinode
multinode: ## NCCL all_reduce and all_gather across the rack over MNNVL
	$(PLAYBOOK) ansible/30-multinode.yml

.PHONY: storage
storage: ## ERASE the local NVMe drives, build /scratch/local, then benchmark it with fio
	$(PLAYBOOK) ansible/20-nvme-raid.yml -e nvme_raid_wipe=true
	$(PLAYBOOK) ansible/25-fio.yml

.PHONY: report
report: ## Write results/report-<ts>.md from the newest smoke and multinode runs
	$(PYTHON) scripts/parse_results.py

.PHONY: down
down: ## Destroy every VM and the IB partition Terraform created
	$(TF) destroy $(TF_APPROVE)
