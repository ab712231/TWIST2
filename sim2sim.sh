SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

# Whole-body policy sim now runs in ISAACSIM (not MuJoCo). This launches the
# IsaacLab low-level runner, which reads the SAME Redis mimic that teleop.sh /
# run_motion_server.sh publish -- so the high-level side is unchanged.
#
# Requires the `isaac` conda env and the GR00T-WBC-Bridge repo (where the IsaacLab
# runner + sim env live). Point WBC_BRIDGE / UNITREE_SIM at your paths.
#   NOTE: run_isaac_wholebody_policy.py's IsaacLab adapter ([WIRE] seams) must be
#   finished on the workstation before this runs end-to-end. See that file + the
#   [V1..V4] checks in gr00t_wbc_bridge/wholebody_policy.py.
WBC_BRIDGE=${WBC_BRIDGE:-$HOME/teleop/GR00T-WBC-Bridge}
UNITREE_SIM=${UNITREE_SIM:-/home/g1/unitree_sim_isaaclab}

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate isaac

cd "$WBC_BRIDGE"
python scripts/run_isaac_wholebody_policy.py \
    --policy "${ckpt_path}" \
    --policy-frequency 100 \
    --unified_usd \
    --unitree_sim_path "${UNITREE_SIM}" \
    --enable_cameras

# --- previous MuJoCo low-level (kept as fallback; `--device cuda` in twist2 env) ---
# cd deploy_real
# python server_low_level_g1_sim.py \
#     --xml ../assets/g1/g1_sim2sim_29dof.xml \
#     --policy ${ckpt_path} \
#     --device cuda \
#     --measure_fps 1 \
#     --policy_frequency 100 \
#     --limit_fps 1
