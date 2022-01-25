import argparse
import os
import random
import habitat
import torch
from arguments import get_args
import numpy as np

from agent.stubborn_agent import QuickAgent






def main():

    #evaluation = "remote"
    args_2 = get_args()
    args_2.sem_gpu_id = 0
    #args_2.device = "cpu" #TODO : only for debug
    #args_2.cuda = 0

    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    #agent = RandomAgent(helper_config=args_2, task_config=config)

    #if args.evaluation == "local":
    #    challenge = habitat.Challenge(eval_remote=False)
    #else:
    #    challenge = habitat.Challenge(eval_remote=True)

    #challenge.submit(agent)
    args_2.num_sem_categories = 23
    nav_agent = QuickAgent(args=args_2,task_config=config)
    #nav_agent = ObjNavAgent_mixed(args=args_2,task_config=config)

    if args_2.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(nav_agent)


if __name__ == "__main__":
    main()
