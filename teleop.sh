# sudo ufw disable

source ~/miniconda3/bin/activate gmr

cd deploy_real

# this is my unitree g1's ip in wifi
# redis_ip="192.168.110.24"
# localhost if you are using laptop to verify sim2sim or sim2real
redis_ip="localhost"

# the height (empirically) should be smaller than the actual human height, due to inaccuracy of the PICO estimation.
actual_human_height=1.6
# Hands are driven by the CONTROLLERS by default: index trigger -> the four
# fingers, joystick -> thumb bend/rotation.
#
# For PICO visual finger tracking (bare hands, no controllers) pass the flag:
#     bash teleop.sh --finger_tracking
# It used to be hardcoded here, which made it impossible to turn off -- `bash
# teleop.sh` with no arguments still ran visual tracking, because the flag was in
# the script rather than on the command line. Anything after `bash teleop.sh` is
# forwarded through "$@" below.
#
# Note the two are mutually exclusive in practice, not in code: the headset
# cannot track bare hands while you are holding the controllers.
python xrobot_teleop_to_robot_w_hand.py --robot unitree_g1 \
             --actual_human_height $actual_human_height \
             --redis_ip $redis_ip \
             --target_fps 100 \
             --measure_fps 1 \
             --hand_type inspire \
             "$@"
            #  --smooth \
            #  --pinch_mode \
