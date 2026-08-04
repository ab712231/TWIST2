"""
Pico 4 Ultra finger tracking → Inspire hand angle conversion.

Converts Pico hand tracking joint positions (OpenXR 26-joint format)
into Inspire RH56DFTP 6-DOF commands (0–1000 per finger).

Inspire DOF mapping:
  Index 0: Pinky       (0=open, 1000=closed)
  Index 1: Ring        (0=open, 1000=closed)
  Index 2: Middle      (0=open, 1000=closed)
  Index 3: Index       (0=open, 1000=closed)
  Index 4: Thumb bend  (0=open, 1000=closed)
  Index 5: Thumb rotation (0=open, 1000=closed)

Pico OpenXR joint indices (26 joints, index 0 = Palm):
  0: Palm
  1: Wrist
  2-5: Thumb (metacarpal, proximal, distal, tip)
  6-10: Index (metacarpal, proximal, intermediate, distal, tip)
  11-15: Middle (metacarpal, proximal, intermediate, distal, tip)
  16-20: Ring (metacarpal, proximal, intermediate, distal, tip)
  21-25: Pinky (metacarpal, proximal, intermediate, distal, tip)
"""

import numpy as np


# Pico OpenXR joint indices per finger
# Each tuple: (metacarpal, proximal, intermediate, distal, tip)
FINGER_JOINTS = {
    'index':  (6, 7, 8, 9, 10),
    'middle': (11, 12, 13, 14, 15),
    'ring':   (16, 17, 18, 19, 20),
    'pinky':  (21, 22, 23, 24, 25),
}

# Thumb joints: (metacarpal, proximal, distal, tip) — no intermediate
THUMB_JOINTS = (2, 3, 4, 5)

# Wrist joint index
WRIST_IDX = 1


def _angle_between_vectors(v1, v2):
    """Compute angle in radians between two 3D vectors.
    
    Returns angle in [0, pi].
    """
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-8 or v2_norm < 1e-8:
        return 0.0
    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def _compute_finger_curl(positions, joint_indices):
    """Compute curl value for a 4-segment finger (index/middle/ring/pinky).
    
    Uses the angle between consecutive bone segments:
      - metacarpal→proximal vs proximal→intermediate
      - proximal→intermediate vs intermediate→distal
      - intermediate→distal vs distal→tip
    
    A straight finger has angles near 0 (π between consecutive vectors
    when measured as deviation from straight). A curled finger has large angles.
    
    Args:
        positions: (N, 3) array of 3D joint positions
        joint_indices: tuple of 5 joint indices (metacarpal, proximal, intermediate, distal, tip)
    
    Returns:
        Curl value in [0.0, 1.0] where 0=straight, 1=fully curled
    """
    meta, prox, inter, dist, tip = joint_indices
    
    p_meta = positions[meta]
    p_prox = positions[prox]
    p_inter = positions[inter]
    p_dist = positions[dist]
    p_tip = positions[tip]
    
    # Bone direction vectors
    v1 = p_prox - p_meta      # metacarpal bone
    v2 = p_inter - p_prox     # proximal phalanx
    v3 = p_dist - p_inter     # intermediate phalanx
    v4 = p_tip - p_dist       # distal phalanx
    
    # Angles between consecutive segments
    # When finger is straight, consecutive vectors point in same direction → angle ≈ 0
    # When finger is curled, vectors diverge → angle increases
    angle1 = _angle_between_vectors(v1, v2)
    angle2 = _angle_between_vectors(v2, v3)
    angle3 = _angle_between_vectors(v3, v4)
    
    # Average angle normalized to [0, 1]
    # Max realistic curl angle per joint is about π/2 (90°)
    # Use π/2 as the normalization factor
    max_angle = np.pi / 2.0
    curl = (angle1 + angle2 + angle3) / (3.0 * max_angle)
    
    return float(np.clip(curl, 0.0, 1.0))


def _compute_thumb_curl(positions):
    """Compute thumb bend (curl) value.
    
    Thumb has fewer joints: metacarpal → proximal → distal → tip.
    
    Args:
        positions: (N, 3) array of 3D joint positions
    
    Returns:
        Curl value in [0.0, 1.0]
    """
    meta, prox, dist, tip = THUMB_JOINTS
    
    p_meta = positions[meta]
    p_prox = positions[prox]
    p_dist = positions[dist]
    p_tip = positions[tip]
    
    # Bone vectors
    v1 = p_prox - p_meta
    v2 = p_dist - p_prox
    v3 = p_tip - p_dist
    
    angle1 = _angle_between_vectors(v1, v2)
    angle2 = _angle_between_vectors(v2, v3)
    
    # Thumb has a wider range of motion, normalize accordingly
    max_angle = np.pi / 2.0
    curl = (angle1 + angle2) / (2.0 * max_angle)
    
    return float(np.clip(curl, 0.0, 1.0))


def _compute_thumb_rotation(positions):
    """Compute thumb rotation (opposition) value.
    
    Measures how much the thumb is rotated toward the palm / other fingers.
    Uses the angle between the thumb direction and the palm plane normal.
    
    Args:
        positions: (N, 3) array of 3D joint positions
    
    Returns:
        Rotation value in [0.0, 1.0] where 0=neutral, 1=full opposition
    """
    wrist = positions[WRIST_IDX]
    index_meta = positions[6]   # Index metacarpal
    pinky_meta = positions[21]  # Pinky metacarpal
    thumb_tip = positions[THUMB_JOINTS[-1]]  # Thumb tip
    thumb_meta = positions[THUMB_JOINTS[0]]  # Thumb metacarpal
    
    # Palm plane: defined by wrist, index metacarpal, pinky metacarpal
    palm_v1 = index_meta - wrist
    palm_v2 = pinky_meta - wrist
    palm_normal = np.cross(palm_v1, palm_v2)
    palm_normal_len = np.linalg.norm(palm_normal)
    if palm_normal_len < 1e-8:
        return 0.0
    palm_normal = palm_normal / palm_normal_len
    
    # Thumb direction vector
    thumb_dir = thumb_tip - thumb_meta
    thumb_dir_len = np.linalg.norm(thumb_dir)
    if thumb_dir_len < 1e-8:
        return 0.0
    thumb_dir = thumb_dir / thumb_dir_len
    
    # Measure how much the thumb points across the palm (toward other fingers)
    # Project thumb direction onto the cross-palm axis
    cross_palm = pinky_meta - index_meta
    cross_palm_len = np.linalg.norm(cross_palm)
    if cross_palm_len < 1e-8:
        return 0.0
    cross_palm = cross_palm / cross_palm_len
    
    # Dot product: positive when thumb is rotating toward pinky side
    rotation = np.dot(thumb_dir, cross_palm)
    
    # Normalize: thumb rotation is roughly [-1, 1], map to [0, 1]
    rotation = float(np.clip((rotation + 1.0) / 2.0, 0.0, 1.0))
    
    return rotation


class _ThumbCalibrator:
    """Rescale the two thumb slots so a real open/closed hand spans the full range.

    The PICO's raw thumb values never reach 0 at full extension -- held-open medians
    measured off the wire were 44 / 159 (left bend/rot) and 141 / 207 (right), out of
    1000. Mapped straight through, the thumb keeps a visible curl no matter how far
    you straighten it, and never quite closes either. The four fingers do not need
    this; only the thumb slots do.

    lo = median of a held fully-open pose (the open hand tracks cleanly).
    hi = the achieved maximum from a full-range sweep, trimmed ~10% so a real full
         curl saturates instead of stopping just short.

    Bounds are FIXED by default. Auto-widening sounds appealing but only ever grows
    the range, so a single dropout frame permanently drags the floor down and
    reinstates the original bug -- enable it only when deliberately recalibrating.
    """

    SLOTS = (4, 5)      # thumb bend, thumb rotation
    MIN_SPAN = 120.0

    SEED = {
        ("left",  4): (44.0, 722.0),    ("right", 4): (141.0, 900.0),
        ("left",  5): (159.0, 644.0),   ("right", 5): (207.0, 668.0),
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

    def span(self, side, slot):
        key = (side, slot)
        if key not in self._lo:
            return 0.0
        return self._hi[key] - self._lo[key]


class _FreshnessMonitor:
    """Detect a hand whose pose feed is frozen even though it reports as tracked.

    get_*_hand_is_active() is a tracking-QUALITY field living inside the same
    per-hand record as the pose, so when acquisition fails the whole record freezes
    -- is_active included. The stale pose is a plausible-looking rest hand, so it
    reads as a deliberate "open hand" and pins the fingers open instead of failing
    loudly.

    Detection is on the raw pose array: N identical consecutive frames means the
    feed is dead, since real tracking always jitters at the millimetre level.
    """

    def __init__(self, stale_frames=12):
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


class PicoFingerTracker:
    """Converts Pico 4 Ultra hand tracking data to Inspire hand commands.
    
    Args:
        smoothing_alpha: EMA smoothing factor (0.0=no smoothing, closer to 1.0=more smoothing).
                        Default 0.3 provides a good balance of responsiveness and stability.
        curl_gain: Multiplier for curl sensitivity. Values > 1.0 make the hand close faster.
        curl_deadzone: Minimum curl value below which output is 0 (reduces jitter at rest).
    """
    
    def __init__(self, smoothing_alpha=0.3, curl_gain=1.5, curl_deadzone=0.05,
                 thumb_calibration=True, thumb_cal_adapt=False, stale_frames=12,
                 freshness=True):
        self.smoothing_alpha = smoothing_alpha
        self.curl_gain = curl_gain
        self.curl_deadzone = curl_deadzone
        # Thumb range correction + frozen-feed guard. Both were validated against
        # the sim before being brought here; see the class docstrings.
        self._thumb_cal = (_ThumbCalibrator(adapt=thumb_cal_adapt)
                           if thumb_calibration else None)
        # Callers that already run their own freshness gate (CorrectedFingerTracker)
        # must disable this one: two gates with different thresholds means the
        # stricter one silently wins and drops hands the caller would have kept.
        self._freshness = _FreshnessMonitor(stale_frames=stale_frames) if freshness else None
        self._warned_frozen = set()
        
        # EMA state per hand
        self._left_smoothed = None
        self._right_smoothed = None
    
    def _apply_ema(self, new_values, smoothed, alpha):
        """Apply exponential moving average smoothing."""
        if smoothed is None:
            return new_values.copy()
        return alpha * smoothed + (1.0 - alpha) * new_values
    
    def _extract_positions(self, hand_data):
        """Extract 3D positions from Pico hand tracking data.
        
        Handles multiple possible data formats:
          - (26, 7) or (27, 7): full OpenXR format with position + quaternion
          - (26, 3) or (27, 3): position-only format
        
        Args:
            hand_data: numpy array of hand tracking data
        
        Returns:
            (N, 3) array of 3D positions, or None if data is invalid
        """
        if hand_data is None:
            return None
        
        hand_data = np.array(hand_data, dtype=np.float64)
        
        if hand_data.ndim != 2:
            return None
        
        rows, cols = hand_data.shape
        
        # Handle (27, 7) format — includes Palm at index 0
        # Pico provides 26 joints (Palm=0, Wrist=1, ..., Pinky_tip=25) 
        # but some APIs add an extra row
        if rows == 27 and cols == 7:
            # Skip Palm (index 0), use indices 1-26 which map to our 0-25
            # Actually keep the original indexing since our constants use 0-based Pico indices
            # where Palm=0, Wrist=1
            return hand_data[:, :3]
        elif rows == 26 and cols == 7:
            # No Palm row — prepend a zero row so our indices work
            palm_row = np.zeros((1, 3))
            return np.vstack([palm_row, hand_data[:, :3]])
        elif rows == 27 and cols == 3:
            return hand_data
        elif rows == 26 and cols == 3:
            palm_row = np.zeros((1, 3))
            return np.vstack([palm_row, hand_data])
        else:
            print(f"[FingerTracker] Unexpected hand data shape: {hand_data.shape}")
            return None
    
    def _curl_to_inspire(self, curl_value):
        """Convert a curl value [0, 1] to Inspire angle [0, 1000].
        
        Applies gain and deadzone.
        """
        if curl_value < self.curl_deadzone:
            return 0.0
        
        # Apply gain and remap
        adjusted = (curl_value - self.curl_deadzone) / (1.0 - self.curl_deadzone)
        adjusted = adjusted * self.curl_gain
        adjusted = np.clip(adjusted, 0.0, 1.0)
        
        return adjusted * 1000.0
    
    def pico_to_inspire_angles(self, hand_data, hand_side='right'):
        """Convert Pico hand tracking data to Inspire hand angles.
        
        Args:
            hand_data: Pico hand tracking array (26×7 or 27×7)
            hand_side: 'left' or 'right'
        
        Returns:
            np.array of 6 values in [0, 1000] range:
              [pinky, ring, middle, index, thumb_bend, thumb_rotation]
            Returns None if hand data is invalid.
        """
        # Frozen-feed guard: a dead feed repeats the same record forever and would
        # otherwise read as a deliberately-held open hand. Returning None makes the
        # caller keep the last good pose rather than snapping the hand open.
        if self._freshness is not None and not self._freshness.is_live(hand_side, hand_data):
            if hand_side not in self._warned_frozen:
                self._warned_frozen.add(hand_side)
                print(f"[finger_tracking][WARN] {hand_side} hand feed frozen "
                      f"({self._freshness.frozen_for(hand_side)} identical frames); "
                      f"holding last pose. Re-acquire tracking to recover.")
            return None
        self._warned_frozen.discard(hand_side)

        positions = self._extract_positions(hand_data)
        if positions is None:
            return None
        
        # Check if all positions are zero (invalid tracking)
        if np.all(np.abs(positions) < 1e-6):
            return None
        
        # Compute per-finger curl
        pinky_curl = _compute_finger_curl(positions, FINGER_JOINTS['pinky'])
        ring_curl = _compute_finger_curl(positions, FINGER_JOINTS['ring'])
        middle_curl = _compute_finger_curl(positions, FINGER_JOINTS['middle'])
        index_curl = _compute_finger_curl(positions, FINGER_JOINTS['index'])
        thumb_curl = _compute_thumb_curl(positions)
        thumb_rot = _compute_thumb_rotation(positions)
        
        # Convert to Inspire range [0, 1000]
        raw_angles = np.array([
            self._curl_to_inspire(pinky_curl),
            self._curl_to_inspire(ring_curl),
            self._curl_to_inspire(middle_curl),
            self._curl_to_inspire(index_curl),
            self._curl_to_inspire(thumb_curl),
            thumb_rot * 1000.0,  # Thumb rotation uses direct mapping
        ], dtype=np.float32)
        
        # Apply EMA smoothing
        if hand_side == 'left':
            self._left_smoothed = self._apply_ema(raw_angles, self._left_smoothed, self.smoothing_alpha)
            result = self._left_smoothed.copy()
        else:
            self._right_smoothed = self._apply_ema(raw_angles, self._right_smoothed, self.smoothing_alpha)
            result = self._right_smoothed.copy()
        
        # Clip to valid range
        result = np.clip(result, 0.0, 1000.0).astype(np.float32)

        # Thumb range correction last, so it operates on the smoothed signal.
        if self._thumb_cal is not None:
            result = self._thumb_cal.apply(hand_side, result)

        return result
    
    def reset(self):
        """Reset smoothing state (call on state transitions)."""
        self._left_smoothed = None
        self._right_smoothed = None
        self._warned_frozen = set()


# ---------------------------------------------------------------------------
# Self-test / dry-run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("PicoFingerTracker - Dry Run Test")
    print("=" * 60)
    
    tracker = PicoFingerTracker(smoothing_alpha=0.0)  # No smoothing for test
    
    # Test 1: Straight hand (all joints in a line along Y axis)
    print("\n--- Test 1: Straight hand (fingers extended) ---")
    positions = np.zeros((27, 7))
    # Wrist at origin
    positions[1, :3] = [0, 0, 0]
    
    # Index finger - straight along Y
    for i, idx in enumerate([6, 7, 8, 9, 10]):
        positions[idx, :3] = [0.04, (i + 1) * 0.02, 0]
    
    # Middle finger
    for i, idx in enumerate([11, 12, 13, 14, 15]):
        positions[idx, :3] = [0.02, (i + 1) * 0.02, 0]
    
    # Ring finger  
    for i, idx in enumerate([16, 17, 18, 19, 20]):
        positions[idx, :3] = [0.0, (i + 1) * 0.02, 0]
    
    # Pinky finger
    for i, idx in enumerate([21, 22, 23, 24, 25]):
        positions[idx, :3] = [-0.02, (i + 1) * 0.02, 0]
    
    # Thumb
    for i, idx in enumerate([2, 3, 4, 5]):
        positions[idx, :3] = [0.06 + i * 0.015, 0.01, 0]
    
    result = tracker.pico_to_inspire_angles(positions, 'right')
    print(f"  Result: {result}")
    print(f"  All in range [0, 1000]: {np.all(result >= 0) and np.all(result <= 1000)}")
    
    # Test 2: Curled hand (fingers curled inward)
    print("\n--- Test 2: Curled hand (fingers closed) ---")
    positions_curled = positions.copy()
    
    # Curl index finger
    positions_curled[8, :3] = [0.04, 0.04, -0.01]   # intermediate bends down
    positions_curled[9, :3] = [0.04, 0.03, -0.025]   # distal curls more
    positions_curled[10, :3] = [0.04, 0.02, -0.03]   # tip curls into palm
    
    # Curl middle
    positions_curled[13, :3] = [0.02, 0.04, -0.01]
    positions_curled[14, :3] = [0.02, 0.03, -0.025]
    positions_curled[15, :3] = [0.02, 0.02, -0.03]
    
    # Curl ring
    positions_curled[18, :3] = [0.0, 0.04, -0.01]
    positions_curled[19, :3] = [0.0, 0.03, -0.025]
    positions_curled[20, :3] = [0.0, 0.02, -0.03]
    
    # Curl pinky
    positions_curled[23, :3] = [-0.02, 0.04, -0.01]
    positions_curled[24, :3] = [-0.02, 0.03, -0.025]
    positions_curled[25, :3] = [-0.02, 0.02, -0.03]
    
    tracker.reset()
    result_curled = tracker.pico_to_inspire_angles(positions_curled, 'right')
    print(f"  Result: {result_curled}")
    print(f"  All in range [0, 1000]: {np.all(result_curled >= 0) and np.all(result_curled <= 1000)}")
    print(f"  Fingers more closed than straight: {np.all(result_curled[:4] > result[0:4])}")
    
    # Test 3: Individual finger curl (only index)
    print("\n--- Test 3: Only index finger curled ---")
    positions_index_only = positions.copy()
    positions_index_only[8, :3] = [0.04, 0.04, -0.01]
    positions_index_only[9, :3] = [0.04, 0.03, -0.025]
    positions_index_only[10, :3] = [0.04, 0.02, -0.03]
    
    tracker.reset()
    result_index = tracker.pico_to_inspire_angles(positions_index_only, 'right')
    print(f"  Result: {result_index}")
    print(f"  Index (idx 3) curled more than others: {result_index[3] > result_index[0] and result_index[3] > result_index[1] and result_index[3] > result_index[2]}")
    
    # Test 4: None / invalid data
    print("\n--- Test 4: Invalid data handling ---")
    tracker.reset()
    result_none = tracker.pico_to_inspire_angles(None, 'right')
    print(f"  None input → {result_none}")
    
    result_zeros = tracker.pico_to_inspire_angles(np.zeros((27, 7)), 'right')
    print(f"  All-zero input → {result_zeros}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
