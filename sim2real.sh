

source ~/miniconda3/bin/activate twist2_deploy

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

# change the network interface name to your own that connects to the robot
net=enp128s31f6

cd deploy_real

# Whole-body policy on the real robot, WITH the real Inspire hands (finger curl
# comes in over Redis action_hand_* from teleop.sh). The real hand has no sim
# velocity cap, so the finger throttling seen in IsaacSim does not apply here.
#
# NECK: run the neck driver separately (it drives the Dynamixel servos):
#   GR00T-WBC-Bridge-teleop/scripts/setup/launch_neck_driver.sh
# Interface (A, recommended): point onboard_neck_driver.py at Redis
#   `action_neck_unitree_g1_with_hands` so neck + body + hands share one bus.
python server_low_level_g1_real.py \
    --policy ${ckpt_path} \
    --net ${net} \
    --device cuda \
    --use_hand \
    --hand_type inspire
    # --smooth_body 0.5
    # --record_proprio
