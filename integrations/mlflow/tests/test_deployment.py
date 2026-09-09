"""Tests for the Crusoe MLflow deployment plugin.

Unit tests use temporary directories and mocked API calls.
Integration tests (marked @pytest.mark.integration) require CRUSOE_API_KEY.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from mlflow.exceptions import MlflowException

from mlflow_crusoe.deployment import CrusoeDeploymentClient


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    """Use a temporary directory for the deployment store."""
    monkeypatch.setenv("MLFLOW_CRUSOE_STORE_DIR", str(tmp_path))
    monkeypatch.setenv("CRUSOE_API_KEY", "test-key")
    return tmp_path


@pytest.fixture
def client(tmp_store):
    return CrusoeDeploymentClient("crusoe")


# ── Create ────────────────────────────────────────────────────────


class TestCreateDeployment:
    def test_create_basic(self, client, tmp_store):
        result = client.create_deployment(
            name="test-deploy",
            model_uri="meta-llama/Llama-3.3-70B-Instruct",
        )
        assert result["name"] == "test-deploy"
        assert result["model"] == "meta-llama/Llama-3.3-70B-Instruct"
        assert (tmp_store / "test-deploy.json").exists()

    def test_create_with_config(self, client, tmp_store):
        client.create_deployment(
            name="custom",
            model_uri="deepseek-ai/DeepSeek-V3",
            config={"temperature": 0.8, "max_tokens": 4096, "top_p": 0.95},
        )
        data = json.loads((tmp_store / "custom.json").read_text())
        assert data["model"] == "deepseek-ai/DeepSeek-V3"
        assert data["temperature"] == 0.8
        assert data["max_tokens"] == 4096
        assert data["top_p"] == 0.95

    def test_create_default_model(self, client, tmp_store):
        client.create_deployment(name="default-model")
        data = json.loads((tmp_store / "default-model.json").read_text())
        assert data["model"] == "meta-llama/Llama-3.3-70B-Instruct"

    def test_create_duplicate_raises(self, client):
        client.create_deployment(name="dup", model_uri="some/model")
        with pytest.raises(MlflowException, match="already exists"):
            client.create_deployment(name="dup", model_uri="some/model")

    def test_create_stop_sequences(self, client, tmp_store):
        client.create_deployment(
            name="with-stop",
            config={"stop": "END,STOP"},
        )
        data = json.loads((tmp_store / "with-stop.json").read_text())
        assert data["stop"] == ["END", "STOP"]


# ── Get / List ────────────────────────────────────────────────────


class TestGetAndList:
    def test_get_deployment(self, client):
        client.create_deployment(name="get-me", model_uri="some/model")
        deployment = client.get_deployment("get-me")
        assert deployment["name"] == "get-me"
        assert deployment["model"] == "some/model"

    def test_get_nonexistent_raises(self, client):
        with pytest.raises(MlflowException, match="not found"):
            client.get_deployment("nope")

    def test_list_empty(self, client):
        assert client.list_deployments() == []

    def test_list_multiple(self, client):
        client.create_deployment(name="a", model_uri="model/a")
        client.create_deployment(name="b", model_uri="model/b")
        deployments = client.list_deployments()
        names = [d["name"] for d in deployments]
        assert "a" in names
        assert "b" in names
        assert len(deployments) == 2


# ── Update ────────────────────────────────────────────────────────


class TestUpdateDeployment:
    def test_update_model(self, client):
        client.create_deployment(name="upd", model_uri="old/model")
        client.update_deployment(name="upd", model_uri="new/model")
        deployment = client.get_deployment("upd")
        assert deployment["model"] == "new/model"

    def test_update_config(self, client):
        client.create_deployment(name="upd2", model_uri="some/model")
        client.update_deployment(
            name="upd2", config={"temperature": 0.9, "max_tokens": 512}
        )
        deployment = client.get_deployment("upd2")
        assert deployment["temperature"] == 0.9
        assert deployment["max_tokens"] == 512

    def test_update_nonexistent_raises(self, client):
        with pytest.raises(MlflowException, match="not found"):
            client.update_deployment(name="nope")


# ── Delete ────────────────────────────────────────────────────────


class TestDeleteDeployment:
    def test_delete(self, client, tmp_store):
        client.create_deployment(name="del-me")
        client.delete_deployment("del-me")
        assert not (tmp_store / "del-me.json").exists()

    def test_delete_idempotent(self, client):
        # Should not raise even if it doesn't exist
        client.delete_deployment("nonexistent")


# ── Predict ───────────────────────────────────────────────────────


class TestPredict:
    def _create_deployment(self, client):
        client.create_deployment(
            name="pred",
            model_uri="meta-llama/Llama-3.3-70B-Instruct",
        )

    def _mock_response(self):
        mock = MagicMock()
        mock.model_dump.return_value = {
            "id": "chatcmpl-test",
            "choices": [
                {"message": {"role": "assistant", "content": "Hello!"}}
            ],
        }
        return mock

    @patch("mlflow_crusoe.deployment.OpenAI")
    def test_predict_with_messages(self, mock_openai_cls, client):
        self._create_deployment(client)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()
        mock_openai_cls.return_value = mock_client

        result = client.predict(
            "pred",
            inputs={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert result["choices"][0]["message"]["content"] == "Hello!"

    @patch("mlflow_crusoe.deployment.OpenAI")
    def test_predict_with_prompt(self, mock_openai_cls, client):
        self._create_deployment(client)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()
        mock_openai_cls.return_value = mock_client

        result = client.predict("pred", inputs={"prompt": "Hi"})
        assert "choices" in result

    @patch("mlflow_crusoe.deployment.OpenAI")
    def test_predict_with_string(self, mock_openai_cls, client):
        self._create_deployment(client)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()
        mock_openai_cls.return_value = mock_client

        result = client.predict("pred", inputs="Hi there")
        assert "choices" in result

    def test_predict_no_name_raises(self, client):
        with pytest.raises(MlflowException, match="deployment_name is required"):
            client.predict(None, inputs="Hi")

    def test_predict_no_inputs_raises(self, client):
        self._create_deployment(client)
        with pytest.raises(MlflowException, match="inputs is required"):
            client.predict("pred", inputs=None)

    def test_predict_bad_inputs_raises(self, client):
        self._create_deployment(client)
        with pytest.raises(MlflowException, match="Unsupported input format"):
            client.predict("pred", inputs=12345)


# ── Input parsing ─────────────────────────────────────────────────


class TestParseInputs:
    def test_parse_messages(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = CrusoeDeploymentClient._parse_inputs({"messages": msgs})
        assert result == msgs

    def test_parse_string(self):
        result = CrusoeDeploymentClient._parse_inputs("Hello")
        assert result == [{"role": "user", "content": "Hello"}]

    def test_parse_prompt_dict(self):
        result = CrusoeDeploymentClient._parse_inputs({"prompt": "Hello"})
        assert result == [{"role": "user", "content": "Hello"}]

    def test_parse_prompt_with_system(self):
        result = CrusoeDeploymentClient._parse_inputs(
            {"prompt": "Hello", "system": "You are helpful."}
        )
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"


# ── Integration tests (require CRUSOE_API_KEY) ───────────────────


@pytest.mark.skipif(
    os.environ.get("CRUSOE_API_KEY") is None
    or os.environ.get("CRUSOE_API_KEY") == "test-key",
    reason="Real CRUSOE_API_KEY not set",
)
class TestIntegration:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MLFLOW_CRUSOE_STORE_DIR", str(tmp_path))

    def test_live_predict(self):
        client = CrusoeDeploymentClient("crusoe")
        client.create_deployment(
            name="live-test",
            model_uri="meta-llama/Llama-3.3-70B-Instruct",
            config={"max_tokens": 32},
        )
        result = client.predict(
            "live-test",
            inputs={"prompt": "What is 2 + 2? Reply with just the number."},
        )
        content = result["choices"][0]["message"]["content"]
        assert "4" in content
