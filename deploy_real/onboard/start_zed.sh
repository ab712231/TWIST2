#!/bin/bash
# Start the ZED Mini streamer on the G1 Orin.
# Usage: bash start_zed.sh [extra args...]
#
# Publishes the left eye as JPEG + aligned uint16 depth (mm) on ZMQ :5556, in the
# same 16-byte-header wire format realsense_streamer.py uses, so both TWIST2's
# VisionClient and the bridge's BinaryCameraClient can read it unchanged.
#
# Ports on the Orin:
#   5555  realsense_streamer.py (D435i, chest, fixed)
#   5556  THIS (ZED Mini, neck-mounted -- moves with the head)
#   5557  mid360_streamer.py (LiDAR)
#
# Calibration: the SDK caches SN<serial>.conf in /usr/local/zed/settings/. If that
# directory is not writable by the user running this, open() fails with the
# misleading "CALIBRATION FILE NOT AVAILABLE". Point it at a writable dir instead:
#   mkdir -p ~/.zed_settings
#   curl -sL -o ~/.zed_settings/SN<serial>.conf \
#        'https://calib.stereolabs.com/?SN=<serial>'
#   export ZED_SETTINGS_PATH=~/.zed_settings/
# (get <serial> from `python3 -c "import pyzed.sl as sl; \
#  print([d.serial_number for d in sl.Camera.get_device_list()])"`)
#
# Depth needs the ZED SDK + pyzed + CUDA -- it is NOT in requirements.txt because
# it cannot be pip-installed. Install the JetPack-matching SDK from stereolabs.com
# on the Orin, then run its get_python_api.py.

export ZED_SETTINGS_PATH="${ZED_SETTINGS_PATH:-$HOME/.zed_settings/}"

python3 ~/g1-onboard/zed_streamer.py \
    --port 5556 \
    --resolution HD720 \
    --fps 30 \
    --depth-mode NEURAL \
    "$@"
