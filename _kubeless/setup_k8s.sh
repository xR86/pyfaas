# install kubeless
export RELEASE=$(curl -s https://api.github.com/repos/kubeless/kubeless/releases/latest | grep tag_name | cut -d '"' -f 4)
kubectl create ns kubeless
kubectl create -f https://github.com/kubeless/kubeless/releases/download/$RELEASE/kubeless-$RELEASE.yaml

# install dashboard 
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v1.10.1/src/deploy/recommended/kubernetes-dashboard.yaml
kubectl create -f https://raw.githubusercontent.com/kubeless/kubeless-ui/master/k8s.yaml


# get cli
# export OS=$(uname -s| tr '[:upper:]' '[:lower:]')
# curl -OL https://github.com/kubeless/kubeless/releases/download/$RELEASE/kubeless_$OS-amd64.zip && unzip kubeless_$OS-amd64.zip && sudo mv bundles/kubeless_$OS-amd64/kubeless /usr/local/bin/
#
# deploy function
echo "ZGVmIGhlbGxvKGV2ZW50LCBjb250ZXh0KToKICBwcmludCBldmVudAogIHJldHVybiBldmVudFsnZGF0YSddCg==" | base64 -d > hello.py
kubeless function deploy hello --runtime python2.7 --from-file hello.py --handler test.hello
kubeless function call hello --data 'Hello TAIP!'

# deploy our services
kubectl apply -f service_accout.yaml
kubectl apply -f pod_scale.yaml
kubectl apply -f pod_prediction.yaml
