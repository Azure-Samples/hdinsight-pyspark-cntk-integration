#! /bin/bash

# Install CNTK on every node. Skip if CNTK latest version is already installed
CNTK_VER="2.0.beta12.0"
CNTK_BASE_URL="https://cntk.ai/PythonWheel/CPU-Only"
CNTK_PY27_WHEEL="cntk-$CNTK_VER-cp27-cp27mu-linux_x86_64.whl"
CNTK_PY35_WHEEL="cntk-$CNTK_VER-cp35-cp35m-linux_x86_64.whl"
ANACONDA_BASEPATH="/usr/bin/anaconda"

# Install prerequisites
sudo apt-get install -y openmpi-bin

check_version_and_install() {
 CNTK_WHEEL=$1
 FIND_PKG=$(pip freeze | grep cntk)
 if [[ $FIND_PKG == "cntk"* ]]; then
   if [[ $FIND_PKG == *"$CNTK_VER" ]]; then
     echo "CNTK latest version is already installed. Skipping..."
   else
     echo "Updating CNTK..."
     pip install --upgrade --no-deps "$CNTK_BASE_URL/$CNTK_WHEEL"
   fi
 else
   echo "Installing CNTK..."
   pip install "$CNTK_BASE_URL/$CNTK_WHEEL"
 fi
}

# Install CNTK in Python 2.7
source "$ANACONDA_BASEPATH/bin/activate"
check_version_and_install $CNTK_PY27_WHEEL

# Install CNTK in Python 3.5
source "$ANACONDA_BASEPATH/bin/activate" py35
check_version_and_install $CNTK_PY35_WHEEL

source "$ANACONDA_BASEPATH/bin/deactivate"

#Check if script action is running on head node. Exit otehrwise.
function get_headnodes
{
    hdfssitepath=/etc/hadoop/conf/hdfs-site.xml
    nn1=$(sed -n '/<name>dfs.namenode.http-address.mycluster.nn1/,/<\/value>/p' $hdfssitepath)
    nn2=$(sed -n '/<name>dfs.namenode.http-address.mycluster.nn2/,/<\/value>/p' $hdfssitepath)

    nn1host=$(sed -n -e 's/.*<value>\(.*\)<\/value>.*/\1/p' <<< $nn1 | cut -d ':' -f 1)
    nn2host=$(sed -n -e 's/.*<value>\(.*\)<\/value>.*/\1/p' <<< $nn2 | cut -d ':' -f 1)

    nn1hostnumber=$(sed -n -e 's/hn\(.*\)-.*/\1/p' <<< $nn1host)
    nn2hostnumber=$(sed -n -e 's/hn\(.*\)-.*/\1/p' <<< $nn2host)

    #only if both headnode hostnames could be retrieved, hostnames will be returned
    #else nothing is returned
    if [[ ! -z $nn1host && ! -z $nn2host ]]
    then
        if (( $nn1hostnumber < $nn2hostnumber )); then
                        echo "$nn1host,$nn2host"
        else
                        echo "$nn2host,$nn1host"
        fi
    fi
}

function get_primary_headnode
{
        headnodes=`get_headnodes`
        echo "`(echo $headnodes | cut -d ',' -f 1)`"
}

PRIMARYHEADNODE=`get_primary_headnode`
fullHostName=$(hostname -f)
if [ "${fullHostName,,}" != "${PRIMARYHEADNODE,,}" ]; then
    echo "$fullHostName is not primary headnode. Skipping ambari config..."
    exit 0
fi

#Constants needed for changing ambari configs
ACTIVEAMBARIHOST=headnodehost
PORT=8080
USERID=$(echo -e "import hdinsight_common.Constants as Constants\nprint Constants.AMBARI_WATCHDOG_USERNAME" | python)
PASSWD=$(echo -e "import hdinsight_common.ClusterManifestParser as ClusterManifestParser\nimport hdinsight_common.Constants as Constants\nimport base64\nbase64pwd = ClusterManifestParser.parse_local_manifest().ambari_users.usersmap[Constants.AMBARI_WATCHDOG_USERNAME].password\nprint base64.b64decode(base64pwd)" | python)
CLUSTERNAME=$(echo -e "import hdinsight_common.ClusterManifestParser as ClusterManifestParser\nprint ClusterManifestParser.parse_local_manifest().deployment.cluster_name" | python)

# Stop and restart affected services
stopServiceViaRest() {
 if [ -z "$1" ]; then
   echo "Need service name to stop service"
   exit 136
 fi
 SERVICENAME=$1
 echo "Stopping $SERVICENAME"
 curl -u "$USERID:$PASSWD" -i -H "X-Requested-By: ambari" -X PUT -d '{"RequestInfo": {"context" :"Stopping Service '"$SERVICENAME"' to install cntk"}, "Body": {"ServiceInfo": {"state": "INSTALLED"}}}' "http://$ACTIVEAMBARIHOST:$PORT/api/v1/clusters/$CLUSTERNAME/services/$SERVICENAME"
}

startServiceViaRest() {
  if [ -z "$1" ]; then
    echo "Need service name to start service"
    exit 136
  fi
  sleep 2
  SERVICENAME="$1"
  echo "Starting $SERVICENAME"
  startResult="$(curl -u $USERID:$PASSWD -i -H 'X-Requested-By: ambari' -X PUT -d '{"RequestInfo": {"context" :"Starting Service '"$SERVICENAME"' with cntk"}, "Body": {"ServiceInfo": {"state": "STARTED"}}}' http://$ACTIVEAMBARIHOST:$PORT/api/v1/clusters/$CLUSTERNAME/services/$SERVICENAME)"
  if [[ "$startResult" == *"500 Server Error"* || "$startResult" == *"internal system exception occurred"* ]]; then
    sleep 60
    echo "Retry starting $SERVICENAME"
    startResult="$(curl -u "$USERID:$PASSWD" -i -H "X-Requested-By: ambari" -X PUT -d '{"RequestInfo": {"context" :"Starting Service '"$SERVICENAME"' with cntk"}, "Body": {"ServiceInfo": {"state": "STARTED"}}}' http://$ACTIVEAMBARIHOST:$PORT/api/v1/clusters/$CLUSTERNAME/services/$SERVICENAME)"
  fi
  echo "$startResult"
}

# Stop affected services service
stopServiceViaRest LIVY
stopServiceViaRest JUPYTER

# Start affected services
startServiceViaRest LIVY
startServiceViaRest JUPYTER
