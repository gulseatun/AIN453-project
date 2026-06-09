#!/bin/bash
source /environment.sh
dt-launchfile-init
rosrun my_package particle_filter_node.py
dt-launchfile-join