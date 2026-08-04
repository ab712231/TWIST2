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

# The Orin usually has NO internet (its default route points at this workstation,
# which does not NAT by default), so pip cannot reach PyPI. The deps are already
# installed there -- split across two interpreters, see start_neck.sh -- so a
# failure here is not fatal. Set SKIP_PIP=1 to skip the attempt entirely.
if [[ "${SKIP_PIP:-0}" == "1" ]]; then
    echo "==> Skipping pip install (SKIP_PIP=1)."
else
    echo "==> Installing Python dependencies on robot (non-fatal if offline)..."
    ssh "${ROBOT_USER}@${ROBOT_IP}" \
        "pip install --no-index --find-links ${REMOTE_DIR}/wheels -r ${REMOTE_DIR}/requirements.txt \
         || pip install -r ${REMOTE_DIR}/requirements.txt" \
        || echo "    WARNING: pip failed (offline?) -- continuing; deps are preinstalled."
fi

echo "==> Deploy complete."
echo
echo "    On the robot:"
echo "      bash ~/g1-onboard/start_realsense.sh   # D435i, chest      -> :5555"
echo "      bash ~/g1-onboard/start_zed.sh         # ZED Mini, neck    -> :5556"
echo "      bash ~/g1-onboard/start_mid360.sh      # MID-360 LiDAR     -> :5557"
echo "      bash ~/g1-onboard/start_neck.sh        # Twist2 neck servos <- Redis"
echo
echo "    On the WORKSTATION (the sender binary is x86-64, it cannot run here):"
echo "      bash deploy_real/start_zed_headset.sh  # ZED ego-view -> PICO headset"
echo
echo "    The ZED needs the ZED SDK + pyzed on the Orin (not pip-installable;"
echo "    see the notes at the top of start_zed.sh)."
echo "    The neck needs serial access:  sudo chmod 777 /dev/ttyUSB0"
