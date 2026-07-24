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

from scipy.spatial.transform import Rotation

from general_motion_retargeting import (
    ROBOT_XML_DICT,
    XRobotStreamer,
)
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from data_utils.finger_tracking import PicoFingerTracker

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None

ROBOT = "unitree_g1"

# Head orientation -> robot z-up frame (same as the relay / pico_streamer). Used
# for the tuned forward-vector neck below.
R_HEADSET_TO_WORLD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])


def _wrap(a):
    """Wrap an angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi

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
    ap.add_argument("--debug-fingers", action="store_true",
                    help="Print hand-tracking state + curl once/sec to diagnose fingers")
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

    if xrt is None:
        print("[bridge] WARNING: xrobotoolkit_sdk not importable -- head tracking "
              "(neck) disabled; arms/fingers still work.")

    left_hand = HAND_OPEN.copy()
    right_hand = HAND_OPEN.copy()
    # Forward-vector neck recenter baseline (captured on first valid head pose).
    neck_yaw0 = None
    neck_pitch0 = None
    dt = 1.0 / max(args.rate_hz, 1.0)
    frames, t_report = 0, time.monotonic()
    try:
        while True:
            t0 = time.monotonic()
            # lhand/rhand from the frame are (is_active, GMR-frame dict) -- unused;
            # fingers are read straight from xrt below (raw Pico-frame arrays).
            smplx_data, _lhand, _rhand, _controller, _headset = streamer.get_current_frame()

            upper_body = None
            if smplx_data is not None:
                qpos = retarget.retarget(smplx_data, offset_to_ground=True)
                left_arm = np.asarray(qpos)[left_arm_idx].astype(np.float32)
                right_arm = np.asarray(qpos)[right_arm_idx].astype(np.float32)

                # Fingers: read the RAW Pico hand-tracking arrays straight from xrt
                # (27x7 = [x,y,z, quat]) -- exactly what pico_to_inspire_angles /
                # _extract_positions were written for, in the raw Pico frame. The
                # streamer's lhand/rhand are (is_active, joint->pose dict) in the
                # coordinate-transformed GMR frame, which the finger tracker cannot
                # consume (np.array on a dict -> TypeError). curl is frame-invariant,
                # so the raw Pico-frame positions are the correct input.
                if finger_tracker is not None and xrt is not None:
                    l_active = bool(xrt.get_left_hand_is_active())
                    r_active = bool(xrt.get_right_hand_is_active())
                    l_state = xrt.get_left_hand_tracking_state()
                    r_state = xrt.get_right_hand_tracking_state()
                    lc = finger_tracker.pico_to_inspire_angles(l_state, "left")
                    rc = finger_tracker.pico_to_inspire_angles(r_state, "right")
                    if args.debug_fingers and frames % max(int(args.rate_hz), 1) == 0:
                        def _summ(s):
                            a = np.asarray(s, dtype=object) if s is not None else None
                            if a is None:
                                return "None"
                            af = np.asarray(s, dtype=np.float64)
                            return f"shape={af.shape} nonzero={int(np.count_nonzero(np.abs(af) > 1e-6))}"
                        print(f"\n[fingers] L active={xrt.get_left_hand_is_active()} "
                              f"raw={_summ(l_state)} "
                              f"curl={'None' if lc is None else np.round(lc, 1)}")
                        print(f"[fingers] R active={xrt.get_right_hand_is_active()} "
                              f"raw={_summ(r_state)} "
                              f"curl={'None' if rc is None else np.round(rc, 1)}")
                    # Only apply when the hand is ACTIVELY tracked. When active=0 the
                    # PICO keeps emitting the last-known (stale) pose, which otherwise
                    # pins the fingers to a frozen curl. Inactive -> hold last command.
                    if l_active and lc is not None:
                        left_hand = curl_to_rad(lc)
                    if r_active and rc is not None:
                        right_hand = curl_to_rad(rc)

                upper_body = np.concatenate(
                    [left_arm, left_hand, right_arm, right_hand]).astype(np.float32)

            # Neck: reuse the relay's TUNED forward-vector head tracking -- decoupled
            # (gimbal-safe) and verified correct in this sim -- rather than TWIST2's
            # neck mapping. Azimuth/elevation of the head forward vector, recentered
            # on the first valid pose. (This is the exact math shipped in the relay.)
            neck = [0.0, 0.0]
            if xrt is not None:
                try:
                    hq = np.array(xrt.get_headset_pose())[3:]  # x, y, z, w
                    if not np.allclose(hq, 0):
                        hr = (R_HEADSET_TO_WORLD
                              @ Rotation.from_quat(hq).as_matrix()
                              @ R_HEADSET_TO_WORLD.T)
                        fwd = hr[:, 0]
                        az = np.arctan2(fwd[1], fwd[0])
                        el = np.arctan2(fwd[2], np.hypot(fwd[0], fwd[1]))
                        if neck_yaw0 is None:
                            neck_yaw0, neck_pitch0 = float(az), float(el)
                        pan = _wrap(float(az) - neck_yaw0) * args.neck_scale
                        # tilt NEGATED: the forward-vector elevation is inverted vs the
                        # sim's neck tilt convention (head up -> neck down otherwise).
                        # Verified regression; see memory pico-neck-decoupled-forward-vector.
                        tilt = -_wrap(float(el) - neck_pitch0) * args.neck_scale
                        neck = [float(np.clip(pan, -3.2, 3.2)),
                                float(np.clip(tilt, -1.6, 1.6))]
                except Exception:
                    neck = [0.0, 0.0]

            # Always publish neck (head tracking is independent of body tracking,
            # like the relay); include the arms/fingers when body data is present.
            msg = {"neck_target": neck}
            if upper_body is not None:
                msg["target_upper_body_pose"] = upper_body
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
