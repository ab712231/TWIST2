#!/bin/bash
# Deploy the onboard sensor streamers + neck driver to the G1 Orin.
# Usage: bash deploy_real/onboard/deploy_to_robot.sh

set -e

ROBOT_USER="unitree"
ROBOT_IP="192.168.123.164"
REMOTE_DIR="~/g1-onboard"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Creating remote directory and copying files to ${ROBOT_USER}@${ROBOT_IP}:${REMOTE_DIR}/..."
ssh "${ROBOT_USER}@${ROBOT_IP}" "mkdir -p ${REMOTE_DIR}"
scp "${SCRIPT_DIR}/realsense_streamer.py" \
    "${SCRIPT_DIR}/start_realsense.sh" \
    "${SCRIPT_DIR}/mid360_streamer.py" \
    "${SCRIPT_DIR}/start_mid360.sh" \
    "${SCRIPT_DIR}/zed_streamer.py" \
    "${SCRIPT_DIR}/start_zed.sh" \
    "${SCRIPT_DIR}/neck_driver.py" \
    "${SCRIPT_DIR}/start_neck.sh" \
    "${SCRIPT_DIR}/requirements.txt" \
    "${ROBOT_USER}@${ROBOT_IP}:${REMOTE_DIR}/"

echo "==> Installing Python dependencies on robot..."
ssh "${ROBOT_USER}@${ROBOT_IP}" "pip install -r ${REMOTE_DIR}/requirements.txt"

echo "==> Deploy complete."
echo
echo "    On the robot:"
echo "      bash ~/g1-onboard/start_realsense.sh   # D435i, chest      -> :5555"
echo "      bash ~/g1-onboard/start_zed.sh         # ZED Mini, neck    -> :5556"
echo "      bash ~/g1-onboard/start_mid360.sh      # MID-360 LiDAR     -> :5557"
echo "      bash ~/g1-onboard/start_neck.sh        # Twist2 neck servos <- Redis"
echo
echo "    The ZED needs the ZED SDK + pyzed on the Orin (not pip-installable;"
echo "    see the notes at the top of start_zed.sh)."
echo "    The neck needs serial access:  sudo chmod 777 /dev/ttyUSB0"
