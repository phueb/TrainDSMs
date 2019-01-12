#!/usr/bin/env bash

/opt/cisco/anyconnect/bin/vpn disconnect
credentials=$(cat ../.vpn_credentials)
/opt/cisco/anyconnect/bin/vpn -s connect vpn.cites.illinois.edu <<< "$credentials"

# sudo mount /media/lab  # TODO no tty present and no askpass program specified

#
cd /home/ph/LudwigCluster/scripts
bash upload_watchers.sh
#
echo "Submitting to Ludwig..."
cd /home/ph/Two_Stage_NLP
source venv/bin/activate
python submit_to_ludwig.py -r2
deactivate
echo "Submission completed"

sleep 5
head -n 6 /media/lab/stdout/*.out