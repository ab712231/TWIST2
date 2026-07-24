"""Bridge: TWIST2 GMR retargeting + visual hand tracking -> GR00T Isaac Lab sim.

Reuses TWIST2's proven front-end -- PICO 5-tracker body pose -> GMR retargeting
-> unitree_g1 joint targets, and the PicoFingerTracker (visual hand -> Inspire) --
and publishes the ARM joints + finger curl + neck to the GR00T Isaac sim's ZMQ
goal port (:5558), in the sim's `target_upper_body_pose` format. This replaces
the crude decoupled_wbc wrist-swap: GMR handles the frames/scale/orientation.

The sim (scripts/run_isaac_sim_loop.py) reads exactly two keys:
  target_upper_body_pose : 26 floats (radians) = left_arm(7) | left_hand(6)
                                                 | right_arm(7) | right_hand(6)
  neck_target            : [pan, tilt]

Run in the `gmr` conda env (needs general_motion_retargeting, mujoco, and TWIST2's
deploy_real on the path). The Isaac sim runs separately in its own env; they talk
over ZMQ, so the env split is fine.

  conda activate gmr
  cd ~/teleop/TWIST2/deploy_real
  python bridge_teleop_to_isaac.py --isaac-host localhost --isaac-port 5558 \
      --actual-human-height 1.6

Activation: this streams continuously; the sim applies whatever it receives. Use
the sim/relay's usual start when you want the robot to follow (or just watch it
track live). Fingers need Hand Tracking ON; arms need the 5 trackers calibrated.
"""
import argparse
import time

import msgpack
import msgpack_numpy as mnp
import mujoco as mj
import numpy as np
import zmq

from general_motion_retargeting import (
    ROBOT_XML_DICT,
    XRobotStreamer,
    human_head_to_robot_neck,
)
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from data_utils.finger_tracking import PicoFingerTracker

ROBOT = "unitree_g1"

# Arm joints in the sim's target_upper_body_pose order (per side).
ARM_SUFFIXES = [
    "shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
    "elbow_joint", "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint",
]

# Inspire hand radian limits (from gr00t_wbc_bridge.g1_inspire_config), order
# [pinky, ring, middle, index, thumb_pitch, thumb_yaw] -- same order PicoFingerTracker
# outputs. curl 0..1 (=0..1000/1000) linearly interpolates open->closed.
HAND_OPEN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1], dtype=np.float32)
HAND_CLOSED = np.array([1.7, 1.7, 1.7, 1.7, 0.5, 1.3], dtype=np.float32)


def build_arm_qpos_index(model):
    """Map each sim arm joint (left_/right_ + ARM_SUFFIXES) to its qpos index in
    GMR's unitree_g1 model. Returns (left_idx[7], right_idx[7]); prints a report
    and raises if any joint name isn't found (so a naming mismatch is obvious)."""
    def lookup(side):
        idx, missing = [], []
        for suf in ARM_SUFFIXES:
            name = f"{side}_{suf}"
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                missing.append(name)
                idx.append(-1)
            else:
                idx.append(int(model.jnt_qposadr[jid]))
        return idx, missing

    l_idx, l_miss = lookup("left")
    r_idx, r_miss = lookup("right")
    print("[bridge] arm joint -> qpos index map:")
    for side, idx in (("left", l_idx), ("right", r_idx)):
        for suf, i in zip(ARM_SUFFIXES, idx):
            print(f"    {side}_{suf}: qpos[{i}]")
    missing = l_miss + r_miss
    if missing:
        # Dump the model's joint names so we can fix the suffixes if needed.
        names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)
                 for j in range(model.njnt)]
        raise SystemExit(
            f"[bridge] joint(s) not found in GMR model: {missing}\n"
            f"  model joints are: {names}\n"
            f"  -> adjust ARM_SUFFIXES / prefixes to match these names.")
    return np.array(l_idx), np.array(r_idx)


def curl_to_rad(pico_angles):
    """PicoFingerTracker 6-vector (0..1000) -> Inspire joint radians (LERP)."""
    curl = np.clip(np.asarray(pico_angles, dtype=np.float32) / 1000.0, 0.0, 1.0)
    return HAND_OPEN + curl * (HAND_CLOSED - HAND_OPEN)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isaac-port", type=int, default=5558,
                    help="ZMQ port to bind; the sim's goal SUB connects to it.")
    ap.add_argument("--actual-human-height", type=float, default=1.6,
                    help="Operator height (m); GMR scales to it (use slightly under).")
    ap.add_argument("--neck-scale", type=float, default=1.0)
    ap.add_argument("--rate-hz", type=float, default=30.0)
    ap.add_argument("--no-fingers", action="store_true", help="Hold hands open")
    args = ap.parse_args()

    # ── GMR retargeting + body/hand streamer (TWIST2's proven front-end) ──
    print("[bridge] initializing GMR retargeting (xrobot -> unitree_g1)...")
    retarget = GMR(src_human="xrobot", tgt_robot=ROBOT,
                   actual_human_height=args.actual_human_height)
    streamer = XRobotStreamer()
    finger_tracker = None if args.no_fingers else PicoFingerTracker()

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[ROBOT]))
    left_arm_idx, right_arm_idx = build_arm_qpos_index(model)

    # ── ZMQ PUB to the sim (mirror the relay: drop stale frames) ──
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 3)
    sock.setsockopt(zmq.LINGER, 0)
    # The sim's goal SUB CONNECTS to the bridge, so the bridge PUB BINDS (same as
    # the relay did). Run the bridge OR the relay -- not both (both bind :5558).
    sock.bind(f"tcp://*:{args.isaac_port}")
    print(f"[bridge] publishing target_upper_body_pose -> bound tcp://*:{args.isaac_port}")

    left_hand = HAND_OPEN.copy()
    right_hand = HAND_OPEN.copy()
    dt = 1.0 / max(args.rate_hz, 1.0)
    frames, t_report = 0, time.monotonic()
    try:
        while True:
            t0 = time.monotonic()
            smplx_data, lhand, rhand, controller_data, headset_data = streamer.get_current_frame()

            upper_body = None
            if smplx_data is not None:
                qpos = retarget.retarget(smplx_data, offset_to_ground=True)
                left_arm = np.asarray(qpos)[left_arm_idx].astype(np.float32)
                right_arm = np.asarray(qpos)[right_arm_idx].astype(np.float32)

                if finger_tracker is not None:
                    lc = finger_tracker.pico_to_inspire_angles(lhand, "left")
                    rc = finger_tracker.pico_to_inspire_angles(rhand, "right")
                    if lc is not None:
                        left_hand = curl_to_rad(lc)
                    if rc is not None:
                        right_hand = curl_to_rad(rc)

                upper_body = np.concatenate(
                    [left_arm, left_hand, right_arm, right_hand]).astype(np.float32)

            # Neck from the head pose (TWIST2's mapping).
            neck = [0.0, 0.0]
            if smplx_data is not None:
                try:
                    ny, npi = human_head_to_robot_neck(smplx_data)
                    neck = [float(ny) * args.neck_scale, float(npi) * args.neck_scale]
                except Exception:
                    neck = [0.0, 0.0]

            if upper_body is not None:
                msg = {"target_upper_body_pose": upper_body, "neck_target": neck}
                sock.send(msgpack.packb(msg, default=mnp.encode), zmq.NOBLOCK)

            frames += 1
            now = time.monotonic()
            if now - t_report >= 2.0:
                state = "tracking" if upper_body is not None else "NO body data"
                print(f"\r[bridge] {state}  {frames/(now-t_report):4.1f} Hz   ",
                      end="", flush=True)
                frames, t_report = 0, now

            time.sleep(max(0.0, dt - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[bridge] stopping.")
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
