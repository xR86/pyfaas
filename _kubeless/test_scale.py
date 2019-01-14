#!/usr/bin/env python3
"""Test the Scale MicroService."""
from unittest.mock import patch, MagicMock
import pytest

import scale

def test_patch_deployment():
    """Test if the patch_deployment actually interacts with Kubernetes."""
    client = MagicMock()
    k8s_client = scale.K8sClient()
    k8s_client._client = client

    k8s_client.patch_deployment("test")
    assert client.patch_namespaced_deployment.call_count == 1

def get_deployment_mock():
    """Get a mocked deployment."""

def test_get_deployments():
    """Test if the patch_deployment actually interacts with Kubernetes."""
    client = MagicMock()
    k8s_client = scale.K8sClient()
    dep = MagicMock()
    dep.metadata.name = None
    scale.DEPLOYMENT = None
    deployments = MagicMock()
    deployments.items = [dep]
    func = MagicMock()
    func.return_value = deployments
    client.list_namespaced_deployment = func
    k8s_client._client = client

    k8s_client.get_deployment()

def test_scale_node_required():
    """Test if when a scale node is required is actually done."""
    deployment = MagicMock()
    deployment.spec.replicas = 5
    func = MagicMock()
    func.return_value = deployment

    client = MagicMock()
    client.get_deployment = func

    scaling = scale.Scaling(client)

    scaling.scale_nodes(100)
    assert func.call_count == 1
    assert client.patch_deployment.call_count == 1
    assert deployment.spec.replicas == 100

def test_get_prediction():
    """Test interaction with prediction server."""
    client = MagicMock()
    http_lib = MagicMock()
    scaling = scale.Scaling(client, http_lib)
    scaling.get_predicted_requests = MagicMock()
    scaling._get_prediction()

    assert scaling.get_predicted_requests.call_count == 1

def test_progress():
    """Test interaction with prediction server."""
    client = MagicMock()
    http_lib = MagicMock()
    scaling = scale.Scaling(client, http_lib)
    scaling._get_prediction = MagicMock()
    scaling.scale_nodes = MagicMock()

    scaling._progress()

    assert scaling.scale_nodes.call_count == 1
    assert scaling._get_prediction.call_count == 1

def test_main_prediction():
    """Test PREDICTION_SERVER validation."""
    scale.PREDICTION_SERVER = None
    with pytest.raises(Exception):
        scale.main()

def test_main_deployment():
    """Test PREDICTION_SERVER validation."""
    scale.PREDICTION_SERVER = "server"
    scale.DEPLOYMENT = None
    with pytest.raises(Exception):
        scale.main()

def test_main_objects():
    """Test PREDICTION_SERVER validation."""
    scale.PREDICTION_SERVER = "server"
    scale.DEPLOYMENT = "deployment"
    scale.K8sClient = MagicMock()
    scale.Scaling = MagicMock()

    scale.main()

    assert scale.K8sClient.call_count == 1
    assert scale.Scaling.call_count == 1

