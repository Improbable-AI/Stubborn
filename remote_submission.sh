#!/usr/bin/env bash

python Stubborn/eval.py --split val --eval 1 --load objnav_pretrained_models/sem_exp.pth --dump_location ./habitat-challenge-data/submit_26/ --print_images 0 --goal all --num_eval_episodes 30 --max_episode_length 100 --num_sem_categories 23 --timestep_limit 451 --no_exp_map 0 --ignore_nongoals 0 --evaluation $AGENT_EVALUATION_TYPE $@
#python Stubborn/mp3d.py --split val --eval 1 --load objnav_pretrained_models/sem_exp.pth --dump_location ./habitat-challenge-data/submit_2/ --print_images 1 --goal all --num_eval_episodes 30 --max_episode_length 150 --num_sem_categories 23 --timestep_limit 10 --online_submission 0 --evaluation $AGENT_EVALUATION_TYPE $@

# 26: 0.8, 201
# 26 night: 0.9, 301
# 25: 0.9, 251
# 27, 12:25: 0.9, 201



# 401, 7200
# 451, 7200
# 351, 7200
