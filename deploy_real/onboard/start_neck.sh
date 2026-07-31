#!/bin/bash
# Start the Twist2 neck driver on the G1 Orin.
# Usage: bash start_neck.sh [extra args...]
#
# Drives the two Dynamixel XC330-T288 servos of the Twist2 neck from the
# [pan, tilt] that teleop.sh publishes to Redis.
#
# WHY THIS RUNS ONBOARD: the servos are on the robot's head. Running the driver
# on the workstation would mean a USB serial cable tethering the head to the
# desk, which is untenable once the robot walks. Same reasoning as the ZED.
#
# REDIS LIVES ON THE WORKSTATION. teleop.sh and sim2real.sh both use localhost
# there, so this process has to reach it across the robot network -- hence
# --redis-host. Override REDIS_HOST if your workstation uses a different IP.
#
# SERIAL PERMISSION: the Dynamixel adapter comes up root-owned. Either run
#   sudo chmod 777 /dev/ttyUSB0
# once per boot, or add the robot user to the dialout group permanently.
#
# CALIBRATION: --pan-gear/--tilt-gear/--*-home/--*-sign defaults are the values
# measured on the physical neck. Verify with --dry-run (prints ticks, moves
# nothing) or --jog (arrow keys) after any mechanical rework.

REDIS_HOST="${REDIS_HOST:-192.168.123.222}"   # workstation on the robot network
DEVICE="${DEVICE:-/dev/ttyUSB0}"

python3 ~/g1-onboard/neck_driver.py \
    --source redis \
    --redis-host "${REDIS_HOST}" \
    --device "${DEVICE}" \
    --rate-hz 50 \
    "$@"
