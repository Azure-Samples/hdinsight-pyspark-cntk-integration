#! /bin/bash

#Installing CNTK on every node
CNTK_HOME="/usr/hdp/current"
ANACONDA_BASEPATH="/usr/bin/anaconda"
cd $CNTK_HOME
curl "https://cntk.ai/binarydrop/CNTK-2-0-beta12-0-Linux-64bit-CPU-Only.tar.gz" | tar xzf -
cd ./cntk/Scripts/install/linux 
./install-cntk.sh --py-version 35 --anaconda-basepath $ANACONDA_BASEPATH

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
AMBARICONFIGS_PY="/var/lib/ambari-server/resources/scripts/configs.py"
ACTIVEAMBARIHOST=headnodehost
PORT=8080
USERID=$(echo -e "import hdinsight_common.Constants as Constants\nprint Constants.AMBARI_WATCHDOG_USERNAME" | python)
PASSWD=$(echo -e "import hdinsight_common.ClusterManifestParser as ClusterManifestParser\nimport hdinsight_common.Constants as Constants\nimport base64\nbase64pwd = ClusterManifestParser.parse_local_manifest().ambari_users.usersmap[Constants.AMBARI_WATCHDOG_USERNAME].password\nprint base64.b64decode(base64pwd)" | python)
CLUSTERNAME=$(echo -e "import hdinsight_common.ClusterManifestParser as ClusterManifestParser\nprint ClusterManifestParser.parse_local_manifest().deployment.cluster_name" | python)
PROTOCOL="HTTP"
ACTION="set"
CONFIG_TYPE="spark2-defaults"
CONFIG_NAME="spark.yarn.appMasterEnv.PYSPARK3_PYTHON"
CONFIG_VALUE="$ANACONDA_BASEPATH/envs/cntk-py35/bin/python"

#Stop and restart affected services
stopServiceViaRest() {
    if [ -z "$1" ]; then
        echo "Need service name to stop service"
        exit 136
    fi
    SERVICENAME=$1
    echo "Stopping $SERVICENAME"
    curl -u $USERID:$PASSWD -i -H 'X-Requested-By: ambari' -X PUT -d '{"RequestInfo": {"context" :"Stopping Service '"$SERVICENAME"' to change Python3 env"}, "Body": {"ServiceInfo": {"state": "INSTALLED"}}}' http://$ACTIVEAMBARIHOST:$PORT/api/v1/clusters/$CLUSTERNAME/services/$SERVICENAME
}

startServiceViaRest() {
    if [ -z "$1" ]; then
        echo "Need service name to start service"
        exit 136
    fi
    sleep 2
    SERVICENAME=$1
    echo "Starting $SERVICENAME"
    startResult=$(curl -u $USERID:$PASSWD -i -H 'X-Requested-By: ambari' -X PUT -d '{"RequestInfo": {"context" :"Starting Service '"$SERVICENAME"' with new Python3 env"}, "Body": {"ServiceInfo": {"state": "STARTED"}}}' http://$ACTIVEAMBARIHOST:$PORT/api/v1/clusters/$CLUSTERNAME/services/$SERVICENAME)
    if [[ $startResult == *"500 Server Error"* || $startResult == *"internal system exception occurred"* ]]; then
        sleep 60
        echo "Retry starting $SERVICENAME"
        startResult=$(curl -u $USERID:$PASSWD -i -H 'X-Requested-By: ambari' -X PUT -d '{"RequestInfo": {"context" :"Starting Service '"$SERVICENAME"' with new Python3 env"}, "Body": {"ServiceInfo": {"state": "STARTED"}}}' http://$ACTIVEAMBARIHOST:$PORT/api/v1/clusters/$CLUSTERNAME/services/$SERVICENAME)
    fi
    echo $startResult
}

#Set new value for Pyspark 3 environment
$AMBARICONFIGS_PY $USERID $PASSWD $PORT $PROTOCOL $ACTION $ACTIVEAMBARIHOST $CLUSTERNAME $CONFIG_TYPE $CONFIG_NAME $CONFIG_VALUE

#Stop affected services service
stopServiceViaRest SPARK2
stopServiceViaRest JUPYTER
stopServiceViaRest LIVY

#Start affected services
startServiceViaRest SPARK2
startServiceViaRest JUPYTER
startServiceViaRest LIVY
