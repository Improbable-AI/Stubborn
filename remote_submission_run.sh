#!/usr/bin/env bash

DOCKER_BUILDKIT=1 docker build . --build-arg INCUBATOR_VER=$(date +%Y%m%d-%H%M%S) --file remote_submission.Dockerfile -t remote_submission
sh remote_submission_objnav_locally_rgbd.sh
#python images/image.py
