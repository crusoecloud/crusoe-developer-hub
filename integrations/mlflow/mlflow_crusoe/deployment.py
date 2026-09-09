"""Crusoe AI deployment client for MLflow.

This plugin enables MLflow to manage model deployments on Crusoe's managed
inference API. Since Crusoe provides a hosted inference service (not a
bring-your-own-model platform), "deployments" represent named endpoint
configurations that route predictions through Crusoe's API.

Usage:
    import mlflow.deployments

    client = mlflow.deployments.get_deploy_client("crusoe")
    client.create_deployment(
        name="my-llm",
        model_uri="meta-llama/Llama-3.3-70B-Instruct",
        config={"temperature": 0.7, "max_tokens": 2048},
    )
    result = client.predict("my-llm", {"prompt": "Hello!"})
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlflow.deployments import BaseDeploymentClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from openai import OpenAI

from mlflow_crusoe.config import (
    DEFAULT_API_BASE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    get_api_key,
)

logger = logging.getLogger(__name__)


def _get_store_path() -> Path:
    """Return the path to the local deployment store."""
    store_dir = Path(
        os.environ.get(
            "MLFLOW_CRUSOE_STORE_DIR",
            os.path.join(Path.home(), ".mlflow", "crusoe_deployments"),
        )
    )
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir


def _deployment_path(name: str) -> Path:
    """Return the file path for a named deployment."""
    return _get_store_path() / f"{name}.json"


def target_help() -> str:
    """Return help text for the Crusoe deployment target."""
    return (
        "The 'crusoe' deployment target routes inference requests to Crusoe AI's\n"
        "managed inference API (https://api.crusoe.ai/v1).\n"
        "\n"
        "Authentication:\n"
        "  Set CRUSOE_API_KEY as an environment variable, or pass\n"
        "  config={'api_key': '...'} when creating a deployment.\n"
        "\n"
        "Supported config keys:\n"
        "  - api_key:           Crusoe API key (overrides env var)\n"
        "  - api_base:          API base URL (default: https://api.crusoe.ai/v1)\n"
        "  - temperature:       Sampling temperature, 0-2 (default: 0.1)\n"
        "  - max_tokens:        Max tokens to generate (default: 1024)\n"
        "  - top_p:             Nucleus sampling parameter\n"
        "  - frequency_penalty: Frequency-based repetition penalty\n"
        "  - presence_penalty:  Presence-based repetition penalty\n"
        "  - stop:              Comma-separated stop sequences\n"
        "\n"
        "The model_uri should be a Crusoe model identifier, e.g.:\n"
        "  meta-llama/Llama-3.3-70B-Instruct\n"
        "  deepseek-ai/DeepSeek-V3\n"
        "  deepseek-ai/DeepSeek-R1\n"
        "  google/gemma-3-12b-it\n"
    )


class CrusoeDeploymentClient(BaseDeploymentClient):
    """MLflow deployment client for Crusoe AI's managed inference API.

    Deployments are stored as local JSON configuration files that map a
    logical name to a Crusoe model + parameters. Predictions are routed
    to Crusoe's OpenAI-compatible API in real time.
    """

    def create_deployment(
        self,
        name: str,
        model_uri: Optional[str] = None,
        flavor: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        endpoint: Optional[str] = None,
    ) -> Dict[str, str]:
        """Create a new Crusoe deployment.

        Args:
            name: Unique deployment name.
            model_uri: Crusoe model identifier
                       (e.g. "meta-llama/Llama-3.3-70B-Instruct").
            flavor: Unused, kept for interface compatibility.
            config: Deployment configuration (temperature, max_tokens, etc.).
            endpoint: Unused, kept for interface compatibility.

        Returns:
            Dict with deployment metadata including 'name' and 'model'.
        """
        config = config or {}
        path = _deployment_path(name)

        if path.exists():
            raise MlflowException(
                f"Deployment '{name}' already exists. Use update_deployment() "
                "to modify it, or delete it first.",
                error_code=RESOURCE_ALREADY_EXISTS,
            )

        model = model_uri or config.pop("model", DEFAULT_MODEL)

        # Validate we can authenticate
        api_key = get_api_key(config)

        deployment = {
            "name": name,
            "model": model,
            "api_base": config.get("api_base", DEFAULT_API_BASE),
            "temperature": float(config.get("temperature", DEFAULT_TEMPERATURE)),
            "max_tokens": int(config.get("max_tokens", DEFAULT_MAX_TOKENS)),
            "created_at": time.time(),
            "updated_at": time.time(),
        }

        # Optional parameters
        for key in ("top_p", "frequency_penalty", "presence_penalty"):
            if key in config:
                deployment[key] = float(config[key])
        if "stop" in config:
            stop = config["stop"]
            deployment["stop"] = (
                stop.split(",") if isinstance(stop, str) else stop
            )

        # Store api_key reference method (never store the key itself)
        if config.get("api_key"):
            deployment["_api_key_source"] = "config"
        else:
            deployment["_api_key_source"] = "env"

        path.write_text(json.dumps(deployment, indent=2))
        logger.info("Created Crusoe deployment '%s' with model '%s'", name, model)

        return {"name": name, "model": model, "flavor": flavor}

    def update_deployment(
        self,
        name: str,
        model_uri: Optional[str] = None,
        flavor: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        endpoint: Optional[str] = None,
    ) -> Dict[str, str]:
        """Update an existing Crusoe deployment.

        Args:
            name: Name of the deployment to update.
            model_uri: New Crusoe model identifier (optional).
            flavor: Unused, kept for interface compatibility.
            config: Updated configuration keys (optional).
            endpoint: Unused, kept for interface compatibility.

        Returns:
            Dict with updated deployment metadata.
        """
        config = config or {}
        deployment = self._load_deployment(name)

        if model_uri:
            deployment["model"] = model_uri

        # Update simple config fields
        updatable_fields = {
            "api_base": str,
            "temperature": float,
            "max_tokens": int,
            "top_p": float,
            "frequency_penalty": float,
            "presence_penalty": float,
        }
        for key, cast_fn in updatable_fields.items():
            if key in config:
                deployment[key] = cast_fn(config[key])

        if "stop" in config:
            stop = config["stop"]
            deployment["stop"] = (
                stop.split(",") if isinstance(stop, str) else stop
            )

        deployment["updated_at"] = time.time()
        _deployment_path(name).write_text(json.dumps(deployment, indent=2))
        logger.info("Updated Crusoe deployment '%s'", name)

        return {"name": name, "model": deployment["model"], "flavor": flavor}

    def delete_deployment(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        """Delete a Crusoe deployment.

        Idempotent — does not raise if the deployment does not exist.

        Args:
            name: Name of the deployment to delete.
            config: Unused, kept for interface compatibility.
            endpoint: Unused, kept for interface compatibility.
        """
        path = _deployment_path(name)
        if path.exists():
            path.unlink()
            logger.info("Deleted Crusoe deployment '%s'", name)
        else:
            logger.info(
                "Deployment '%s' does not exist; nothing to delete.", name
            )

    def list_deployments(
        self,
        endpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all Crusoe deployments.

        Returns:
            List of deployment dicts, each containing at least 'name'.
        """
        store = _get_store_path()
        deployments = []
        for path in sorted(store.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                deployments.append(data)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping corrupt deployment file %s: %s", path, exc)
        return deployments

    def get_deployment(
        self,
        name: str,
        endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get details of a specific Crusoe deployment.

        Args:
            name: Deployment name.

        Returns:
            Dict with full deployment configuration.
        """
        return self._load_deployment(name)

    def predict(
        self,
        deployment_name: Optional[str] = None,
        inputs: Optional[Any] = None,
        endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run inference against a Crusoe deployment.

        Supports two input formats:

        Chat format (recommended):
            {"messages": [{"role": "user", "content": "Hello!"}]}

        Simple prompt format:
            {"prompt": "Hello!"}
            or just a string: "Hello!"

        Args:
            deployment_name: Name of the deployment to query.
            inputs: Prediction input (dict or string).
            endpoint: Unused, kept for interface compatibility.

        Returns:
            Dict with the API response.
        """
        if deployment_name is None:
            raise MlflowException(
                "deployment_name is required.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        deployment = self._load_deployment(deployment_name)
        api_key = get_api_key()

        client = OpenAI(
            api_key=api_key,
            base_url=deployment.get("api_base", DEFAULT_API_BASE),
        )

        # Build messages from inputs
        messages = self._parse_inputs(inputs)

        # Build API parameters
        params: Dict[str, Any] = {
            "model": deployment["model"],
            "messages": messages,
            "temperature": deployment.get("temperature", DEFAULT_TEMPERATURE),
            "max_tokens": deployment.get("max_tokens", DEFAULT_MAX_TOKENS),
        }
        for key in ("top_p", "frequency_penalty", "presence_penalty", "stop"):
            if key in deployment:
                params[key] = deployment[key]

        response = client.chat.completions.create(**params)
        return response.model_dump()

    # ── Private helpers ───────────────────────────────────────────

    def _load_deployment(self, name: str) -> Dict[str, Any]:
        """Load a deployment from the local store."""
        path = _deployment_path(name)
        if not path.exists():
            raise MlflowException(
                f"Deployment '{name}' not found. Use list_deployments() to see "
                "available deployments.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return json.loads(path.read_text())

    @staticmethod
    def _parse_inputs(inputs: Any) -> List[Dict[str, str]]:
        """Normalize various input formats into OpenAI-style messages."""
        if inputs is None:
            raise MlflowException(
                "inputs is required for predict().",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Already a messages list
        if isinstance(inputs, dict) and "messages" in inputs:
            return inputs["messages"]

        # Simple prompt string
        if isinstance(inputs, str):
            return [{"role": "user", "content": inputs}]

        # Dict with "prompt" key
        if isinstance(inputs, dict) and "prompt" in inputs:
            messages = []
            if "system" in inputs:
                messages.append({"role": "system", "content": inputs["system"]})
            messages.append({"role": "user", "content": inputs["prompt"]})
            return messages

        raise MlflowException(
            "Unsupported input format. Use one of:\n"
            '  {"messages": [{"role": "user", "content": "..."}]}\n'
            '  {"prompt": "..."}\n'
            '  "plain string"',
            error_code=INVALID_PARAMETER_VALUE,
        )
