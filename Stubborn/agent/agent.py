import argparse
import os
import random
import habitat
import torch
from arguments import get_args
import numpy as np
import agent.utils.pose as pu
from constants import coco_categories, hab2coco, hab2name, habitat_labels_r, fourty221, fourty221_ori, habitat_goal_label_to_similar_coco
import copy
from agent.quick_agent_states import Quick_Agent_State
from agent.agent_helper import Quick_Sem_Exp_Env_Agent_Helper
from agent.utils.object_identification import get_prediction



class QuickAgent(habitat.Agent):
    def __init__(self,args,task_config: habitat.Config):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.agent_states = Quick_Agent_State(args)
        if args.collision_module == 0:
            self.agent_helper = Quick_Sem_Exp_Env_Agent_Helper(args,self.agent_states)
        elif args.collision_module == 1:
            self.agent_helper = Quick_Sem_Exp_Env_Agent_Helper_1(args,
                                                               self.agent_states)
        elif args.collision_module == 2:
            self.agent_helper = Quick_Sem_Exp_Env_Agent_Helper_2(args,
                                                               self.agent_states)
        elif args.collision_module == 3:
            self.agent_helper = Quick_Sem_Exp_Env_Agent_Helper_3(args,
                                                               self.agent_states)
        self.last_sim_location = None
        self.device = args.device
        self.first_obs = True
        self.valid_goals = 0
        self.total_episodes = 0
        self.args = args
        self.timestep = 0
        self.low_score_threshold = 0.7
        self.high_score_threshold = 0.9
        # towel tv shower gym clothes
        self.low_score_categories = {13,14,15,19,21}
    def reset(self):
        self.agent_helper.reset()
        self.agent_states.reset()
        self.last_sim_location = None
        self.first_obs = True
        self.step = 0
        self.timestep = 0
        self.total_episodes += 1
        #self.agent_helper.set_small_collision_map() # TODO: you probably don't want this!!!

    def act(self, observations):
        self.timestep += 1
        if self.timestep > self.args.timestep_limit and self.agent_states.found_goal == False:
            return {'action': 0}
        if self.timestep > 495:
            return {'action': 0}
        #get first preprocess
        goal = observations['objectgoal']
        goal = goal[0]+1
        if self.args.remote_only_one_cat != -1 and goal != self.args.remote_only_one_cat:
            return {'action':0}
        if self.args.only6 != 0:
            if not (goal in hab2coco.keys()):
                return {'action':0}
        if self.args.threshold_mode == 0 or ( self.args.threshold_mode == 1 and goal in self.low_score_categories):
            self.agent_states.score_threshold = self.low_score_threshold
        if self.args.threshold_mode == 2:
            self.agent_states.score_threshold = self.high_score_threshold
        info = self.get_info(observations)

        # get second preprocess
        self.agent_helper.set_goal_cat(goal)
        obs, info = self.agent_helper.preprocess_inputs(observations['rgb'],observations['depth'],info)
        if self.args.num_sem_categories == 23:
            info['goal_cat_id'] = goal
            info['goal_name'] = habitat_labels_r[goal]

        info = [info]
        obs = obs[np.newaxis,:,:,:]
        # now ready to be passed to agent states
        obs = torch.from_numpy(obs).float().to(self.device)
        if self.first_obs:
            self.agent_states.init_with_obs(obs,info)
            self.first_obs = False


        planner_inputs = self.agent_states.upd_agent_state(obs,info)
        # now get action
        action = self.agent_helper.plan_act_and_preprocess(planner_inputs)
        #return {'action':1}
        #TODO: For real submission, make sure to disable this
        if self.args.no_stop == 1 and action['action'] == 0:
            print('219 shouldnt happen')
            self.agent_states.clear_goal_and_set_gt_map(planner_inputs[0]['goal'])
            return {'action':1}
        if action['action'] == 0 and self.args.goal_selection_scheme == 0:
            item = self.agent_states.goal_record(planner_inputs[0]['goal'])
            stp = get_prediction(item,goal)
            if stp:
                return action
            else:
                self.agent_states.clear_goal(
                    planner_inputs[0]['goal'])
                return {'action': 1}
        return action

    def get_info(self, obs):

        info = {}
        dx, dy, do = self.get_pose_change(obs)
        info['sensor_pose'] = [dx, dy, do]
        # set goal
        return info

    def get_sim_location(self,obs):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        #agent_state = super().habitat_env.sim.get_agent_state(0)
        #x = -agent_state.position[2]
        #y = -agent_state.position[0]

        nap2 = obs['gps'][0]
        nap0 = -obs['gps'][1]
        x = nap2
        y = nap0
        o = obs['compass']
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self,obs):
        curr_sim_pose = self.get_sim_location(obs)
        if self.last_sim_location is not None:
            dx, dy, do = pu.get_rel_pose_change(
                curr_sim_pose, self.last_sim_location)
            dx,dy,do = dx[0],dy[0],do[0]
        else:
            dx, dy, do = 0,0,0
        self.last_sim_location = curr_sim_pose
        return dx, dy, do





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
