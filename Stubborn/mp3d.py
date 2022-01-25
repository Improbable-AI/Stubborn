from agent.stubborn_agent import QuickAgent
import argparse
import os
import random
import habitat
import torch
from arguments import get_args
import numpy as np
from constants import twentyone240
from agent.utils.object_identification import recal_predictors

episodes = set()


class HabitatEnv:
    def __init__(self,args,agent,cfg):
        self.agent = agent
        self.args = args
        self.env = habitat.Env(
            config=habitat.get_config(cfg))
        self.suc = 0

    def run_for(self,num_episodes,start = 0):
        ps = None
        s = None
        for i in range(num_episodes):
            explored_object = False
            step_of_finding = -1
            observations = self.env.reset()
            s = self.env.sim.habitat_config.SCENE
            if self.args.exclude_current_scene == 1:
                if ps is None or s != ps:
                    recal_predictors((i, i + 199))
            ps = s

            scene = self.env._sim.semantic_scene
            instance_id_to_label_id = {
                int(obj.id.split("_")[-1]): (obj.category.index(),obj.category.name()) for obj in
                scene.objects}
            obj_40_id = twentyone240[observations['objectgoal'][0]+1] + 1
            #print(observations['objectgoal'][0]+1)
            #print(observations['objectgoal'])
            #print(observations)
            self.agent.reset()
            if len(episodes) != 0:
                if (i+1) not in episodes:
                    continue
            if i < start:
                continue

            # Step through environment with random actions
            epi_len = 0
            while not self.env.episode_over:
                a = np.unique(observations['semantic'])
                if explored_object == False:
                    for id in a:
                        obj_id, name = instance_id_to_label_id[id]
                        if obj_id == obj_40_id:
                            if np.sum(observations['semantic'] == id) > 2500:
                                explored_object = True
                                step_of_finding = epi_len
                                break
                if self.args.use_gt_mask == 1:
                    mask = np.zeros(observations['semantic'].shape)
                    for id in a:
                        obj_id, name = instance_id_to_label_id[id]
                        if obj_id == obj_40_id:
                            mask[observations['semantic'] == id] = 1.0
                    if self.args.chaplot == 0:
                        self.agent.agent_helper.sem_pred_rednet.set_gt_mask(mask)
                if self.args.chaplot_gt == 1 and self.args.chaplot == 1:
                    mask = np.zeros(observations['semantic'].shape)
                    if explored_object:
                        for id in a:
                            obj_id, name = instance_id_to_label_id[id]
                            if obj_id == obj_40_id:
                                mask[observations['semantic'] == id] = 1.0
                    self.agent.agent_helper.sem_pred_rednet.set_gt_mask(
                        mask,observations['objectgoal'][0]+1)

                action = self.agent.act(observations)
                if action['action'] == 0 and self.args.chaplot == 1 and self.args.chaplot_forward == 1:
                    self.env.step({'action':1})

                observations = self.env.step(action)
                epi_len += 1
                if self.args.early_stop and explored_object:
                    break

            #print(observations)
            #print(self.env.get_metrics())
            self.suc += self.env.get_metrics()["success"]
            #if self.args.chaplot == 0:
            self.agent.agent_states.save_conf_stat(self.env.get_metrics()["success"],epi_len,i,explored_object,step_of_finding)

            print(i,self.suc,self.env.get_metrics())





def main():
    task_cfg_path = "Stubborn/envs/habitat/configs/tasks/objectnav_mp3d.yaml"

    args_2 = get_args()
    args_2.sem_gpu_id = 0
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    args_2.num_sem_categories = 23
    if args_2.chaplot == 0:
        nav_agent = QuickAgent(args=args_2,task_config=config)
    else:
        print("unsupported agent type")
        exit(0)
    env = HabitatEnv(args_2,nav_agent,task_cfg_path)
    env.run_for(1,start = 0)




if __name__ == "__main__":
    main()
