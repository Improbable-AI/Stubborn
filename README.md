# Stubborn: A Strong Baseline for Habitat ObjectNav Challenge
This is the implementation of the Stubborn Agent (yuumi_the_magic_cat on the Habitat Challenge Leaderboard).
It uses relatively simple strategies but achieved a strong result (0.237 success rate and 0.098 SPL).
We release the code here to help the Robotic community by providing them with a strong baseline that they can both compete against or work from.


### Overview:
The Stubborn Agent consists of 4 modules: a mapping module, a global goal module, a path planning module,
and a multi-frame object detection module. The mapping module builds map of obstacles and goal objects over time, the global goal module gives a long term goal for the agent to reach, the path planning
module plans the actual path to reach the goal, and the goal detection module determines whether the agent has reached the goal object.

The primary code contributions from the paper are located in:

TODO: describe where in the source code each modules are located in.


## Requirements

We use Docker to run the code, therefore users don't need to manually install any dependencies.
Users need to download pretrained weights and environment dataset before they can run the code.

### Pretrained Weights

Users can download pretrained weights for RedNet [here](https://drive.google.com/drive/folders/1SM75RweHtHQ13lu9fZkVjkOlWMaWpFuZ?usp=sharing).

Users can download data used to train the Object Detection Module [here](add thi link)


### Downloading scene dataset
- Download the Matterport 3D dataset using the instructions here: https://niessner.github.io/Matterport/
- Move the Matterport 3D scene dataset or create a symlink at `data/scene_datasets/mp3d`.

### Setting up datasets
The code requires the datasets in a `data` folder in the following format (same as habitat-lab):
```
Object-Goal-Navigation/
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
TODO: add a script that build docker and run it
```

The pre-trained model should get 0.267 Success, 0.136 SPL and 4.121 DTG.


## Cite as
TODO: make our own

### Bibtex:
TODO

## Related Projects
- This project builds on the [Goal-Oriented Semantic Policy](https://devendrachaplot.github.io/projects/semantic-exploration) paper.
- Rednet Object Segmentation Model is trained by Joel Ye et. al at https://joel99.github.io/objectnav/.
