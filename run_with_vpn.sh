#!/usr/bin/env bash

#/opt/cisco/anyconnect/bin/vpn disconnect
#credentials=$(cat ../.vpn_credentials)
#/opt/cisco/anyconnect/bin/vpn -s connect vpn.cites.illinois.edu <<< "$credentials"
#
#pwd=$(cat /home/ph/.sudo_pwd)
#echo ${pwd} | sudo -S mount /media/lab
#
#cd /home/ph/LudwigCluster/scripts
#bash upload_watchers.sh

echo "Submitting to Ludwig..."
cd /home/ph/Two_Stage_NLP
source venv/bin/activate
python submit_to_ludwig.py -r2 -s -w hinton
deactivate
echo "Submission completed"

sleep 5
head -n 6 /media/lab/stdout/*.out