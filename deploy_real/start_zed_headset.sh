#!/bin/bash
# Stream the ZED ego-view to the PICO headset (H.264 over TCP).
# Usage: bash deploy_real/start_zed_headset.sh [extra OrinVideoSender args...]
#
# RUNS ON THE WORKSTATION, not the robot. Despite the name, OrinVideoSender is a
# ZMQ *subscriber* -- its own default is --zmq-sub tcp://192.168.123.164:5555,
# i.e. it already expects to pull frames off the robot across the network. The
# prebuilt binary is x86-64 and the Orin is aarch64, so it cannot run there at
# all without being rebuilt on the Orin.
#
# It also happens to be the only arrangement that reaches the headset: the robot
# sits on the isolated 192.168.123.x wired link, while the PICO is on WiFi with
# this workstation. The sender bridges the two.
#
#     robot :5556  --(wired 192.168.123.x)-->  THIS  --(WiFi H.264)-->  PICO
#
# It parses the same 16-byte header zed_streamer.py emits:
#     [width(4)][height(4)][rgb_jpeg_len(4)][depth_data_len(4)][JPEG][depth]
# (see main_d435_zmq.cpp:542). Depth is ignored; RGB+depth frames pass fine.
#
# Because it only subscribes, the camera is still opened ONCE by zed_streamer.py
# and fanned out -- so the operator can see through the robot WHILE episodes are
# being recorded. These are not mutually exclusive.
#
# Port 5556 is the ZED; the binary's built-in default is 5555 (the RealSense),
# hence the explicit --zmq-sub below.
#
# Ports (from the sender's README): 13579 = listen/control, 12345 = video send.
# LISTEN MODE NEEDS NO HEADSET IP -- the headset connects in and sends OPEN_CAMERA.
#
# Prereqs: start_zed.sh already running on the robot.

SENDER="${SENDER:-$HOME/XRoboToolkit-Orin-Video-Sender/OrinVideoSender}"
ROBOT_IP="${ROBOT_IP:-192.168.123.164}"
# Prefer the stereo side-car when the streamer is publishing one. The headset's
# ZEDMINI profile asks for 2560x720 side-by-side; feeding it the mono :5556 frame
# gets it stretched across a stereo rect and it does not fill the FOV. :5558 only
# exists when start_zed.sh was given --stereo-port, so fall back rather than hang
# on a dead endpoint (a ZMQ SUB never errors -- it just waits forever).
STEREO_PORT="${STEREO_PORT:-5558}"
if [[ -z "${ZMQ_SUB:-}" ]]; then
    if timeout 1 bash -c "echo > /dev/tcp/${ROBOT_IP}/${STEREO_PORT}" 2>/dev/null; then
        ZMQ_SUB="tcp://${ROBOT_IP}:${STEREO_PORT}"
        echo "Found stereo stream on :${STEREO_PORT} -- using it (full-FOV binocular view)."
    else
        ZMQ_SUB="tcp://${ROBOT_IP}:5556"
        echo "No stereo stream on :${STEREO_PORT} -- falling back to mono :5556."
        echo "  For the full-FOV view restart the streamer with:"
        echo "    bash ~/g1-onboard/start_zed.sh --stereo-port ${STEREO_PORT}"
    fi
fi
HEADSET_IP="${HEADSET_IP:-}"                  # only needed for push mode
# The headset asks for 4 Mbps via OPEN_CAMERA (its video_source.yml), which is
# thin for a 2560x720 stereo pair and shows as blocking on motion. Raising it
# here avoids needing adb to edit the yml on the headset. 0 = honour the request.
BITRATE="${BITRATE:-10000000}"

if [[ ! -x "$SENDER" ]]; then
    echo "ERROR: OrinVideoSender not found at $SENDER"
    echo "  Set SENDER=/path/to/OrinVideoSender, or build it:"
    echo "    cd ~/XRoboToolkit-Orin-Video-Sender && make"
    exit 1
fi

echo "ZED ego-view: subscribing to ${ZMQ_SUB}"

if [[ -z "$HEADSET_IP" ]]; then
    echo "Listening on 0.0.0.0:13579 -- open XRoboToolkit on the headset and connect here."
    exec "$SENDER" --zmq-sub "$ZMQ_SUB" --bitrate "$BITRATE" --listen 0.0.0.0:13579 "$@"
else
    echo "Pushing to headset ${HEADSET_IP}:12345"
    exec "$SENDER" --zmq-sub "$ZMQ_SUB" --bitrate "$BITRATE" --send --server "$HEADSET_IP" --port 12345 "$@"
fi
