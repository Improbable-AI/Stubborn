# Stubborn: A Strong Baseline for Habitat ObjectNav Challenge
This is the implementation of the Stubborn Agent (yuumi_the_magic_cat on the Habitat Challenge Leaderboard).
It uses relatively simple strategies but achieved a strong result (0.237 success rate and 0.098 SPL).
We release the code here to help the Robotic community by providing them with a strong baseline that they can both compete against or work from.


### Overview:
The Stubborn Agent is modified from [Goal-Oriented-Semantic-Policy](https://devendrachaplot.github.io/projects/semantic-exploration) and makes improvement in exploration strategy, untrapping strategy, and object detection strategy, which are discueed in details in the paper.

![example](./Stubborn/docs/demo.gif)


The core of the code in located in `Stubborn/agent`.

`Stubborn/agent/stubborn_agent.py` contains the implementation of the stubborn agent;

`Stubborn/agent/agent_state.py` contains implementation of the global goal module and goal detection module;

`Stubborn/agent/mapping_module.py` contains implementation of the mapping module;

`Stubborn/agent/agent_helper.py` contains implementation of the path planning module.



## Requirements

We use Docker to run the code, therefore users don't need to manually install any dependencies.
Users need to download pretrained weights and environment dataset before they can run the code.

### Pretrained Weights

Users can download pretrained weights for RedNet [here](https://drive.google.com/drive/folders/1SM75RweHtHQ13lu9fZkVjkOlWMaWpFuZ?usp=sharing).
Download the `rednet_semmap_mp3d_tuned.pth` file and place it in `Stubborn`.


Users can download data used to train the Object Detection Module [here](https://drive.google.com/file/d/1M8ArawCSD-91pNvxTvXmXKMwKUFFHKOJ/view?usp=sharing).
Place `obj_id_data.pickle` into the `Stubborn` folder.


### Downloading scene dataset
- Download the Matterport 3D dataset using the instructions [here](https://niessner.github.io/Matterport/)
- Move the Matterport 3D scene dataset or create a symlink at `data/scene_datasets/mp3d`.

- Download the Matterport 3D Object Navigation Task Data [here](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip).
- Move the task dataset to `data/datasets`

- The data folder should look like this:
```
Stubborn/
  data/
    scene_datasets/
      mp3d/
        17DRP5sb8fy
        1LXtFkjw3qL
        ...
    datasets/
      objectnav/
        mp3d/
          v1/
            train/
            val/
            val_mini/
```


### Test setup
To verify that the data is setup correctly, run:
```
sh remote_submission_run.sh
```

The agent should get 0.233 Success, 0.139 SPL and 4.301 DTG.




## Related Projects
- This project builds on the [Goal-Oriented Semantic Policy](https://devendrachaplot.github.io/projects/semantic-exploration) paper.
- Rednet Object Segmentation Model is trained by Joel Ye et. al at https://joel99.github.io/objectnav/.

