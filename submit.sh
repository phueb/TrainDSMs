#!/usr/bin/env bash


cd /home/ph/LudwigCluster/scripts
bash kill_job.sh 2StageNLP
bash reload_watcher.sh

cd /home/ph/Two_Stage_NLP
source venv/bin/activate
python submit.py -r2 -s -x
deactivate

sleep 5
tail -n 10 /media/lab/stdout/*.out