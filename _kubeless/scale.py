#!/usr/bin/env python
"""Scaling microservice."""
from __future__ import print_function

import os
import time
import json
import math

import requests

from kubernetes import client, config

from aspect_logging import CallLoggingAspect

# pylint: disable=broad-except, too-few-public-methods

PREDICTION_SERVER = os.getenv("PYFAAS_PREDICTION_SERVER", None)
DEPLOYMENT = os.getenv("PYFAAS_DEPLOYMENT", None)
NAMESPACE = os.getenv("PYFAAS_NAMESPACE", 'default')
SCALE_DELAY = int(os.getenv("PYFAAS_SCALE_DELAY", '5'))
LOOP_SLEEP = 2
REQUESTS_PER_CONTAINER = 10

IN_CLUSTER = os.path.isdir("/var/run/secrets/kubernetes.io")
if IN_CLUSTER:
    config.load_incluster_config()
else:
    config.load_kube_config()


class K8sClient():
    """Adaptor for the Kubernetes API."""
    def __init__(self):
        self._client = client.ExtensionsV1beta1Api()

    @CallLoggingAspect("Query deployments", "")
    def get_deployment(self):
        """Get the Kubernetes deployment."""
        deployments = self._client.list_namespaced_deployment(namespace=NAMESPACE)
        for deployment in deployments.items:
            if deployment.metadata.name == DEPLOYMENT:
                return deployment
        raise Exception("Deployment {} was not found".format(DEPLOYMENT))

    @CallLoggingAspect("Patch deployment ...", "")
    def patch_deployment(self, deployment):
        """Patch the Kubernetes deployment."""
        status = self._client.patch_namespaced_deployment(
            name=DEPLOYMENT,
            namespace="default",
            body=deployment)
        return status.status


class Scaling():
    """Scaling reconciliation loop."""

    def __init__(self, k8s_client, http_lib=requests):
        self._k8s_client = k8s_client
        self._http_lib = http_lib

    @staticmethod
    def get_predicted_requests(response):
        """Parse a requests."""
        data = json.loads(response.text)
        return int(data['predicted'])

    @CallLoggingAspect("Take scale decision", "Scaling decision done")
    def scale_nodes(self, containers_count):
        """Scale the number of containers."""
        deployment = self._k8s_client.get_deployment()
        if deployment.spec.replicas != containers_count:
            print("[ DECISION ]Scale from {} -> {} ".format(
                deployment.spec.replicas, containers_count))
            deployment.spec.replicas = containers_count
            time.sleep(SCALE_DELAY)
        else:
            print("[ DECISION ] We already have {} replicas ".format(containers_count))
        self._k8s_client.patch_deployment(deployment)


    @CallLoggingAspect("Query the prediction server", "Querying Done")
    def _get_prediction(self):
        """Get the predicted number of container."""
        response = self._http_lib.get(PREDICTION_SERVER)
        response.raise_for_status()

        predicted_requests = self.get_predicted_requests(response)
        nr_of_containers = math.ceil(predicted_requests / float(REQUESTS_PER_CONTAINER))

        return nr_of_containers

    @CallLoggingAspect("Evaluate one loop iteration ", "")
    def _progress(self):
        """Take a step in the reconciliation loop."""
        nr_of_containers = self._get_prediction()
        self.scale_nodes(nr_of_containers)

    def start(self):
        """Start the reconciliation loop."""
        while True:
            try:
                self._progress()
            except Exception as exc:
                print("Got excpetion :", exc)
            time.sleep(LOOP_SLEEP)

def main():
    """Main entry point."""
    if PREDICTION_SERVER is None:
        raise Exception('PYFAAS_PREDICTION_SERVER must be set as environment variable.')

    if DEPLOYMENT is None:
        raise Exception('PYFAAS_DEPLOYMENT must be set as environment variable.')

    k8s_client = K8sClient()
    scaling = Scaling(k8s_client, requests)
    scaling.start()

if __name__ == '__main__':
    main()
