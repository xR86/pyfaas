#!/usr/bin/env python
from __future__ import print_function

import os
import time
import json
import math

import requests

from kubernetes import client, config

PREDICTION_SERVER = os.getenv("PYFAAS_PREDICTION_SERVER", None)
DEPLOYMENT = os.getenv("PYFAAS_DEPLOYMENT", None)
NAMESPACE = os.getenv("PYFAAS_NAMESPACE", 'default')
SCALE_DELAY = int(os.getenv("PYFAAS_SCALE_DELAY", '5'))
LOOP_SLEEP = 2
REQUESTS_PER_CONTAINER = 10


IN_CLUSTER = os.path.isdir("/var/run/secrets/kubernetes.io")
if IN_CLUSTER:
    config.load_incluster_config
else:
    config.load_kube_config()


def get_predicted_requests(response):
    """Parse a requests."""
    data = json.loads(response.text)
    return int(data['predicted'])


def get_deployment():
    """Get the Kubernetes deployment."""
    api = client.ExtensionsV1beta1Api()
    deployments = api.list_namespaced_deployment(namespace=NAMESPACE)
    for deployment in deployments.items:
        if deployment.metadata.name == DEPLOYMENT:
            return deployment
    raise Exception("Deployment {} was not found".format(DEPLOYMENT))


def patch_deployment(deployment):
    """Patch the Kubernetes deployment."""
    api = client.ExtensionsV1beta1Api()
    status = api.patch_namespaced_deployment(
        name=DEPLOYMENT,
        namespace="default",
        body=deployment)
    return status.status


def scale_nodes(containers_count):
    """Scale the number of containers."""
    deployment = get_deployment()
    if deployment.spec.replicas != containers_count:
        print("Scale from {} -> {} ".format(deployment.spec.replicas, containers_count))
        deployment.spec.replicas = containers_count
        time.sleep(SCALE_DELAY)
    else:
        print("We already have {} replicas ".format(containers_count))
    resp = patch_deployment(deployment)
    # print("Scale done :", resp.)

def main():
    """Main entry point."""
    if PREDICTION_SERVER is None:
        raise Exception('PYFAAS_PREDICTION_SERVER must be set as environment variable.')

    if DEPLOYMENT is None:
        raise Exception('DEPLOYMENT must be set as environment variable.')

    while True:
        try:
            print("Evaluation ... ")
            response = requests.get(PREDICTION_SERVER)
            response.raise_for_status()

            predicted_requests = get_predicted_requests(response)
            nr_of_containers = math.ceil(predicted_requests / float(REQUESTS_PER_CONTAINER))

            scale_nodes(nr_of_containers)
        except Exception as exc:
            print("Got excpetion :", exc)
        time.sleep(LOOP_SLEEP)
if __name__ == '__main__':
    main()
