#!/usr/bin/env bash


cd /home/ph/LudwigCluster/scripts
bash kill_job.sh 2ProcessNLP
bash reload_watcher.sh

cd /home/ph/Two_Stage_NLP
source venv/bin/activate
python submit.py -r3 -x
deactivate

sleep 5
tail -n 10 /media/lab/stdout/*.out