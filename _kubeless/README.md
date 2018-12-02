# How to install Kubeless

First of all [kubeless](kubeless) is only a project that relies on [Kubernetes](kubernetes) to actually run functions.

1. Install [Kubernetes](kubernetes) locally. An easy way is to use the [Minikube](minikube) project.
2. Install the [Kubeless][kubeless] project following [this](kubeless-install) instructions
```bash
export RELEASE=$(curl -s https://api.github.com/repos/kubeless/kubeless/releases/latest | grep tag_name | cut -d '"' -f 4)
kubectl create ns kubeless
kubectl create -f https://github.com/kubeless/kubeless/releases/download/$RELEASE/kubeless-$RELEASE.yaml
```
3. Get the CLI
```bash
export OS=$(uname -s| tr '[:upper:]' '[:lower:]')
curl -OL https://github.com/kubeless/kubeless/releases/download/$RELEASE/kubeless_$OS-amd64.zip && unzip kubeless_$OS-amd64.zip && sudo mv bundles/kubeless_$OS-amd64/kubeless /usr/local/bin/
```
4. Deploy a sample function
```bash
echo "ZGVmIGhlbGxvKGV2ZW50LCBjb250ZXh0KToKICBwcmludCBldmVudAogIHJldHVybiBldmVudFsnZGF0YSddCg==" | base64 -d > hello.py
kubeless function deploy hello --runtime python2.7 --from-file hello.py --handler test.hello
```
4. Call the function (you may need to wait for the pod to actually start)
```bash
kubeless function call hello --data 'Hello TAIP!'
```

[kubeless]: https://kubeless.io/
[kubernetes]: https://kubernetes.io/
[minikube]: https://kubernetes.io/docs/tasks/tools/install-minikube/
[kubeless-install]: https://kubeless.io/docs/quick-start/
