from agent import QuickAgent
import argparse
import os
import random
import habitat
import torch
from arguments import get_args
import numpy as np
from constants import twentyone240
from object_identification import recal_predictors

episodes = set()
#episodes = {19,45}
#episodes = {19} # this is where blocking was suspected
#episodes = {1,201,402,600,797}
#episodes = {201,402,600,797}
#episodes = {2}
#episodes = {201}
#for i in range(61,500,20):
#    episodes.add(i)
#print(episodes)
#episodes = {81}
#episodes = {91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 551, 552, 553, 554, 555, 556, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 1057, 1058, 1059, 1060, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1697, 1698, 1699, 1700, 1752, 1753, 1754, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 110, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 931, 932, 933, 934, 935, 936, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1600, 1601, 1602, 1603, 1604, 1605, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1966, 1967, 1968, 1969, 547, 548, 549, 550, 1584, 1920, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 1713, 1714, 1715, 1716, 1404, 1405, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1873, 1874, 1875, 1876}
#episodes = {547, 60, 61,  1713, 1714, }
#episodes = {1,11,21,31,41,51,61,71,81,91,101,111,121,131,141,151,161,171,181,191}
#episodes = {1,101,191}
#episodes = {31,6,7,8}
#episodes = {110,111,112,113,}
#episodes = {1,2,3,4,5,6,7}
#episodes = {7}

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

            print(i,self.suc)





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
    env.run_for(2,start = 0)




if __name__ == "__main__":
    main()
