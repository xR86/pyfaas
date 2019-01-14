kubectl proxy -p 8888 &
echo $!
echo "Start load"
while true; 
do
    RESP=$(curl -s -L --data '{"Another": "Echo"}' --header "Content-Type:application/json" localhost:8888/api/v1/namespaces/default/services/hello:http-function-port/proxy/)
    echo $RESP
    sleep 1
done
