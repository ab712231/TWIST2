"""Shared PICO finger + neck overlays (the operator's tuned versions).

Extracted verbatim from bridge_teleop_to_isaac.py so BOTH the bridge (ZMQ ->
Isaac upper-body) and xrobot_teleop_to_robot_w_hand.py (Redis -> RL whole-body)
apply the SAME corrected finger pipeline and forward-vector neck. This is what
makes teleop.sh carry the operator's fixes into the whole-body flow.

Fingers -- the four fixes, in order:
  1. read the RAW xrt hand arrays directly (the streamer hands back a
     (is_active, dict) tuple that crashes pico_to_inspire_angles);
  2. _align_hand_state: pad 26 -> 27 so _extract_positions doesn't shift the
     joint indices (open hand was reading as closed);
  3. _FreshnessMonitor: gate on the pose feed actually CHANGING, since the PICO
     reports is_active=1 for an untracked hand and returns a frozen rest pose;
  4. _ThumbCalibrator: rescale the thumb channels to full travel per hand.
Output is the 0..1000 Inspire curl (what the real Inspire hand + Redis want);
use curl_to_rad() for a sim that needs radian joint targets.

Neck -- NeckForwardVector: decoupled (gimbal-safe) azimuth/elevation of the head
forward vector, recentered on first pose, tilt NEGATED (head up -> neck up).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from data_utils.finger_tracking import PicoFingerTracker

# Inspire radian limits, order [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
HAND_OPEN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1], dtype=np.float32)
HAND_CLOSED = np.array([1.7, 1.7, 1.7, 1.7, 0.5, 1.3], dtype=np.float32)

# Head orientation -> robot z-up frame.
R_HEADSET_TO_WORLD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])


def _wrap(a):
    """Wrap an angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def curl_to_rad(pico_angles):
    """Inspire 6-vector (0..1000) -> joint radians (LERP open->closed)."""
    curl = np.clip(np.asarray(pico_angles, dtype=np.float32) / 1000.0, 0.0, 1.0)
    return HAND_OPEN + curl * (HAND_CLOSED - HAND_OPEN)


def _align_hand_state(state):
    """Pad xrt's 26-joint hand array to 27 so _extract_positions keeps indices
    2-25 aligned (its 26-row branch prepends a palm and shifts everything +1)."""
    if state is None:
        return None
    a = np.asarray(state, dtype=np.float64)
    if a.ndim == 2 and a.shape[0] == 26:
        a = np.vstack([a, np.zeros((1, a.shape[1]))])
    return a


class _FreshnessMonitor:
    """Detect a hand whose pose feed is frozen despite is_active reporting 1.

    The PICO reports is_active=1 for a hand it is not tracking and returns a
    constant rest pose (curls 0, thumb rotation ~130) -- indistinguishable from a
    real open hand by value, so it pins the fingers open. Live tracking always
    jitters at float precision, so bit-identical consecutive frames mean dead feed.
    """

    def __init__(self, stale_frames):
        self._stale_frames = max(int(stale_frames), 3)
        self._prev = {}
        self._count = {}

    def is_live(self, side, state):
        prev = self._prev.get(side)
        self._prev[side] = None if state is None else np.array(state, copy=True)
        if state is None:
            self._count[side] = 0
            return False
        if prev is None or prev.shape != np.asarray(state).shape:
            self._count[side] = 0
            return True
        if np.array_equal(prev, state):
            self._count[side] = self._count.get(side, 0) + 1
        else:
            self._count[side] = 0
        return self._count[side] < self._stale_frames

    def frozen_for(self, side):
        return self._count.get(side, 0)


class _ThumbCalibrator:
    """Rescale the thumb channels to full travel per hand (they only cover
    ~65..740 of 1000 raw, and differ left vs right). Seeded from measured held
    endpoints; does not latch onto a running min (a closed-fist dropout would drag
    the floor down). See bridge_teleop_to_isaac.py for the full rationale."""

    SLOTS = (4, 5)
    MIN_SPAN = 120.0
    SEED = {
        ("left", 4): (44.0, 722.0),  ("right", 4): (141.0, 900.0),
        ("left", 5): (159.0, 644.0), ("right", 5): (207.0, 668.0),
    }

    def __init__(self, seed=True, adapt=False):
        self._lo = {}
        self._hi = {}
        self._adapt = adapt
        if seed:
            for key, (lo, hi) in self.SEED.items():
                self._lo[key], self._hi[key] = lo, hi

    def apply(self, side, curl):
        if curl is None:
            return None
        out = np.array(curl, dtype=np.float32, copy=True)
        for s in self.SLOTS:
            key = (side, s)
            v = float(out[s])
            if key not in self._lo or self._adapt:
                self._lo[key] = v if key not in self._lo else min(self._lo[key], v)
                self._hi[key] = v if key not in self._hi else max(self._hi[key], v)
            lo, hi = self._lo[key], self._hi[key]
            if hi - lo >= self.MIN_SPAN:
                out[s] = float(np.clip((v - lo) / (hi - lo) * 1000.0, 0.0, 1000.0))
        return out


class CorrectedFingerTracker:
    """The operator's full corrected finger pipeline as one call.

    get_curls(xrt) -> (left6, right6), each the 0..1000 Inspire curl for a hand
    that is actively tracked AND whose feed is live, else None (caller HOLDS its
    last commanded curl). Mirrors the bridge loop exactly.
    """

    def __init__(self, rate_hz=30.0, thumb_cal=True, thumb_adapt=False):
        self.tracker = PicoFingerTracker()
        self.freshness = _FreshnessMonitor(stale_frames=rate_hz * 0.4)
        self.thumb_cal = _ThumbCalibrator(adapt=thumb_adapt) if thumb_cal else None

    def get_curls(self, xrt):
        out = {}
        sides = (
            ("left", xrt.get_left_hand_is_active, xrt.get_left_hand_tracking_state),
            ("right", xrt.get_right_hand_is_active, xrt.get_right_hand_tracking_state),
        )
        for side, active_fn, state_fn in sides:
            active = bool(active_fn())
            state = _align_hand_state(state_fn())
            curl = self.tracker.pico_to_inspire_angles(state, side)
            live = self.freshness.is_live(side, state)
            if curl is not None and self.thumb_cal is not None:
                curl = self.thumb_cal.apply(side, curl)
            out[side] = curl if (active and live and curl is not None) else None
        return out["left"], out["right"]


class NeckForwardVector:
    """Decoupled forward-vector head->neck (pan, tilt), recentered on first pose.
    gimbal-safe (no Euler cross-coupling).

    Sign convention matches GR00T-WBC-Bridge's run_teleop_inspire_relay.py, which is
    the version validated against the physical Twist2 neck. Tilt is NOT negated here:
    onboard_neck_driver.py applies the head->servo direction itself via --tilt-sign
    (default -1). Negating in both places cancels out and drives the neck the wrong
    way, which is what this producer used to do."""

    def __init__(self, neck_scale=1.0, pan_clip=3.2, tilt_clip=1.6):
        self.neck_scale = neck_scale
        self.pan_clip = pan_clip
        self.tilt_clip = tilt_clip
        self._yaw0 = None
        self._pitch0 = None

    def compute(self, head_quat_xyzw):
        """head_quat_xyzw: xrt.get_headset_pose()[3:] (x, y, z, w). -> [pan, tilt]."""
        hq = np.asarray(head_quat_xyzw, dtype=float)
        if hq.shape[0] != 4 or np.allclose(hq, 0):
            return [0.0, 0.0]
        hr = R_HEADSET_TO_WORLD @ Rotation.from_quat(hq).as_matrix() @ R_HEADSET_TO_WORLD.T
        fwd = hr[:, 0]
        az = np.arctan2(fwd[1], fwd[0])
        el = np.arctan2(fwd[2], np.hypot(fwd[0], fwd[1]))
        if self._yaw0 is None:
            self._yaw0, self._pitch0 = float(az), float(el)
        pan = _wrap(float(az) - self._yaw0) * self.neck_scale
        tilt = _wrap(float(el) - self._pitch0) * self.neck_scale
        return [float(np.clip(pan, -self.pan_clip, self.pan_clip)),
                float(np.clip(tilt, -self.tilt_clip, self.tilt_clip))]
