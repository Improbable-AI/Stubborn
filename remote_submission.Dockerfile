FROM fairembodied/habitat-challenge:testing_2021_habitat_base_docker

RUN /bin/bash -c ". activate habitat"

# Install project specific packages
RUN /bin/bash -c "apt-get update; apt-get install -y libsm6 libxext6 libxrender-dev; . activate habitat; pip install opencv-python"
RUN /bin/bash -c ". activate habitat; pip install --upgrade cython numpy"
RUN /bin/bash -c ". activate habitat; pip install matplotlib seaborn==0.9.0 scikit-fmm==2019.1.30 scikit-image==0.15.0 imageio==2.6.0 scikit-learn==0.22.2.post1 ifcfg"

# Install pytorch and torch_scatter
RUN conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.2 -c pytorch

# Install detectron2
RUN /bin/bash -c ". activate habitat; python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html"

RUN /bin/bash -c ". activate habitat; pip install torchvision==0.7.0 "

RUN /bin/bash -c ". activate habitat; pip install gym==0.10.5 "
ARG INCUBATOR_VER=unknown
ADD remote_submission.sh remote_submission.sh
ADD configs/challenge_objectnav2021.local.rgbd.yaml /challenge_objectnav2021.local.rgbd.yaml
ENV AGENT_EVALUATION_TYPE remote
ADD Stubborn Stubborn
ENV PYTHONPATH "${PYTHONPATH}:/Stubborn"


ENV TRACK_CONFIG_FILE "/challenge_objectnav2021.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash remote_submission.sh"]
