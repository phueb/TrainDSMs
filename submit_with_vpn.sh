#!/usr/bin/env bash


cd /home/ph/LudwigCluster/scripts
#bash reload_watcher.sh
bash kill_job.sh 2StageNLP

cd /home/ph/Two_Stage_NLP
source venv/bin/activate
python submit.py -r3 -s
deactivate

sleep 5
tail -n 6 /media/lab/stdout/*.out