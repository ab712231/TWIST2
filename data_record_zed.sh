#!/bin/bash
# Data recording with the neck-mounted ZED Mini (RGB + aligned depth).
# Usage: bash data_record_zed.sh
#
# Requires the ZED streamer running on the Orin:
#   ssh unitree@192.168.123.164
#   bash ~/g1-onboard/start_zed.sh          # publishes RGB+depth on :5556
#
# Replaces the old ZED recording path (server_data_record.py + OrinVideoSender
# on :5555, stereo RGB, no depth).

source ~/miniconda3/bin/activate twist2

cd deploy_real

robot_ip="192.168.123.164"
data_frequency=30

python server_data_record_zed.py \
    --frequency ${data_frequency} \
    --robot_ip ${robot_ip} \
    --goal "pick up the red cup" \
    --desc "A humanoid robot picks up a red cup from the table." \
    --steps "step1: approach table. step2: grasp cup. step3: lift cup."
