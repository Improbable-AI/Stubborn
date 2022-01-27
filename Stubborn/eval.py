import argparse
import os
import random
import habitat
import torch
from arguments import get_args
import numpy as np

from agent.stubborn_agent import StubbornAgent

def main():

    args_2 = get_args()
    args_2.sem_gpu_id = 0
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    nav_agent = StubbornAgent(args=args_2,task_config=config)
    if args_2.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
    else:
        challenge = habitat.Challenge(eval_remote=True)
    challenge.submit(nav_agent)


if __name__ == "__main__":
    main()
