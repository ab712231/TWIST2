"""Onboard neck driver: relay neck_target -> Twist2 Dynamixel servos.

This is the physical-robot analog of run_isaac_sim_loop.py's apply_goal() neck
block. Instead of writing pan/tilt into a sim articulation, it drives the two
Dynamixel servos of the Twist2 neck.

Wire contract (identical to the sim):
  * run_teleop_inspire_relay.py PUBs the full goal dict on ZMQ :5558 (msgpack).
  * We SUB-connect to that same socket (alongside the sim, if any), pull
    `neck_target = [pan, tilt]` out of each message, and actuate.
  * All the head-tracking math (recenter, wrap, sign, scale) already lives in
    the relay, so this script is a dumb actuator: target radians -> servo ticks.

Twist2 servo setup (from TWIST2/doc/TWIST2_NECK.md):
  * yaw (pan)  motor -> Dynamixel ID 0
  * pitch(tilt)motor -> Dynamixel ID 1
  * servos: Dynamixel XC330-T288 (X-series, Protocol 2.0)
  * baud 2 Mbps, port /dev/ttyUSB0
  * 4096 ticks/rev at the OUTPUT shaft -- the servo's internal 288.35:1 gearbox
    is already baked into that, so tick<->angle needs no gear factor. --pan-gear/
    --tilt-gear are ONLY for any EXTERNAL horn->head reduction (direct drive = 1.0).

Hardware-specific values you MUST verify for your build are exposed as CLI args
(gear ratio, direction sign, home tick, limits). The defaults assume a 1:1
direct-drive neck centered at tick 2048 -- measure yours and override.

Usage (after the relay is running):
  sudo chmod 777 /dev/ttyUSB0
  python scripts/onboard_neck_driver.py
  python scripts/onboard_neck_driver.py --pan-sign -1 --pan-gear 2.0 --extended
"""
import argparse
import sys
import time

import numpy as np

# msgpack / msgpack_numpy / zmq are imported lazily inside the --source zmq
# branch. They are only needed to decode the sim relay's goal dict; the real
# deployment path (--source redis) never touches them. Importing them here made
# the driver unrunnable on the G1 Orin, which has neither the packages nor
# internet access to fetch them -- and msgpack ships C extensions, so it cannot
# be satisfied by copying an x86 wheel across either.

try:
    from dynamixel_sdk import (
        COMM_SUCCESS,
        GroupSyncWrite,
        PacketHandler,
        PortHandler,
    )
except ImportError:
    sys.exit(
        "dynamixel_sdk is required for the onboard neck driver.\n"
        "  pip install dynamixel-sdk\n"
        "and make sure the U2D2/servo bus is on /dev/ttyUSB0 (sudo chmod 777 /dev/ttyUSB0)."
    )

# ── Dynamixel Protocol 2.0 X-series control table (model-agnostic) ───────────
PROTOCOL_VERSION = 2.0
ADDR_HARDWARE_ERROR = 70   # 1 byte: latched hardware-error bits
ADDR_ID = 7                # 1 byte (EEPROM; torque must be off to write)
ADDR_BAUD_RATE = 8         # 1 byte enum (EEPROM)
ADDR_OPERATING_MODE = 11   # 1 byte: 3 = position, 4 = extended (multi-turn)
# Protocol 2.0 X-series baud enum (value written to ADDR_BAUD_RATE)
BAUD_ENUM = {9600: 0, 57600: 1, 115200: 2, 1000000: 3, 2000000: 4, 3000000: 5, 4000000: 6}
ADDR_TORQUE_ENABLE = 64    # 1 byte
ADDR_GOAL_POSITION = 116   # 4 bytes
ADDR_PRESENT_POSITION = 132  # 4 bytes
LEN_GOAL_POSITION = 4
TORQUE_ENABLE, TORQUE_DISABLE = 1, 0
OP_MODE_POSITION, OP_MODE_EXTENDED = 3, 4
TICKS_PER_REV = 4096
TICKS_PER_RAD = TICKS_PER_REV / (2.0 * np.pi)

# Measured on the physical Twist2 neck. Home = "look straight ahead" tick; the
# min/max ticks are the mechanical stops with a small safety margin. Clamping in
# ABSOLUTE tick space (not a symmetric rad limit) keeps the full asymmetric range
# -- e.g. tilt travels much further forward/down than back -- and is immune to
# sign flips. Measured stops: pan 72/4045, tilt 2276/3667.
PAN_HOME_DEFAULT = 1956
TILT_HOME_DEFAULT = 2724
PAN_MIN_TICK, PAN_MAX_TICK = 115, 4005      # from stops 72 / 4045 (+~40 margin)
TILT_MIN_TICK, TILT_MAX_TICK = 2292, 3646   # from stops 2286 / 3652 (+~6 margin)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # ── relay ZMQ source (neck_target lives in the goal dict on :5558) ──
    p.add_argument("--host", default="localhost", help="Host the relay PUBs from")
    p.add_argument("--port", type=int, default=5558, help="Relay goal ZMQ port")
    p.add_argument("--rate-hz", type=float, default=50.0, help="Servo write rate (Hz)")
    p.add_argument("--dropout-hold-s", type=float, default=0.5,
                   help="Hold last target this long after the stream drops; then re-home.")
    # ── neck source: Redis (real deployment) or ZMQ (sim relay/bridge) ──
    p.add_argument("--source", choices=("redis", "zmq"), default="redis",
                   help="Where neck_target comes from. 'redis' (default, real deployment): "
                        "reads the [pan,tilt] list teleop publishes to action_neck_* on Redis, "
                        "same bus as the body policy. 'zmq': the sim relay/bridge goal dict on :5558.")
    p.add_argument("--redis-host", default="localhost", help="Redis host (--source redis)")
    p.add_argument("--redis-port", type=int, default=6379, help="Redis port (--source redis)")
    p.add_argument("--neck-key", default="action_neck_unitree_g1_with_hands",
                   help="Redis key holding [pan, tilt] (--source redis)")
    # ── Dynamixel bus ──
    p.add_argument("--device", default="/dev/ttyUSB0", help="Serial device")
    p.add_argument("--baud", type=int, default=2000000, help="Baud rate (TWIST2: 2 Mbps)")
    p.add_argument("--yaw-id", type=int, default=0, help="Pan/yaw servo ID (TWIST2: 0)")
    p.add_argument("--pitch-id", type=int, default=1, help="Tilt/pitch servo ID (TWIST2: 1)")
    p.add_argument("--extended", action="store_true",
                   help="Use extended-position (multi-turn) mode instead of single-turn.")
    # ── per-axis calibration (MEASURE THESE FOR YOUR NECK) ──
    p.add_argument("--pan-min", type=int, default=PAN_MIN_TICK, help="Pan lower tick stop")
    p.add_argument("--pan-max", type=int, default=PAN_MAX_TICK, help="Pan upper tick stop")
    p.add_argument("--tilt-min", type=int, default=TILT_MIN_TICK, help="Tilt lower tick stop")
    p.add_argument("--tilt-max", type=int, default=TILT_MAX_TICK, help="Tilt upper tick stop")
    p.add_argument("--pan-gear", type=float, default=1.0,
                   help="EXTERNAL horn->head ratio for pan (1.0 = direct drive; the "
                        "servo's internal 288.35:1 is already in its 4096 ticks/rev)")
    p.add_argument("--tilt-gear", type=float, default=1.0,
                   help="EXTERNAL horn->head ratio for tilt (1.0 = direct drive)")
    p.add_argument("--pan-scale", type=float, default=1.0,
                   help="Head-to-neck gain for pan (>1 amplifies; relay sends 1:1)")
    p.add_argument("--tilt-scale", type=float, default=1.0,
                   help="Head-to-neck gain for tilt (1.0 = accurate 1:1 with your head; "
                        ">1 amplifies but breaks 1:1)")
    p.add_argument("--pan-sign", type=int, default=-1, choices=(-1, 1),
                   help="Flip if the head pans the wrong way (measured: -1)")
    p.add_argument("--tilt-sign", type=int, default=-1, choices=(-1, 1),
                   help="Flip if the head tilts the wrong way (measured: -1 vs the "
                        "forward-vector relay; +1 was inverted, masked by the old tilt clip)")
    p.add_argument("--pan-home", type=int, default=PAN_HOME_DEFAULT,
                   help="Servo tick at pan=0 / look-straight-ahead (measured: 1956)")
    p.add_argument("--tilt-home", type=int, default=TILT_HOME_DEFAULT,
                   help="Servo tick at tilt=0 / level gaze (measured: 2724)")
    p.add_argument("--hold-torque-on-exit", action="store_true",
                   help="Leave torque enabled on exit (default: disable so the neck goes limp).")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't open the serial bus or move servos; just print the ticks "
                        "each neck_target maps to. Use to check the mapping before energizing.")
    p.add_argument("--calibrate", action="store_true",
                   help="Torque OFF both servos and live-print present position; move the "
                        "head by hand to read home tick, limits, and direction. No ZMQ. "
                        "(Only works on back-drivable axes -- use --jog for geared ones.)")
    p.add_argument("--scan", action="store_true",
                   help="Broadcast-ping across common baud rates to find every servo's "
                        "actual ID + baud. Use when a servo 'is not responding'.")
    p.add_argument("--configure", action="store_true",
                   help="Set ONE connected servo's ID and baud (connect one at a time). "
                        "Use with --current-id/--current-baud/--new-id/--new-baud.")
    p.add_argument("--current-id", type=int, default=1, help="--configure: servo's present ID (factory=1)")
    p.add_argument("--current-baud", type=int, default=57600, help="--configure: servo's present baud (factory=57600)")
    p.add_argument("--new-id", type=int, default=None, help="--configure: ID to write")
    p.add_argument("--new-baud", type=int, default=2000000, help="--configure: baud to write (default 2 Mbps)")
    p.add_argument("--jog", action="store_true",
                   help="Torque ON; arrow keys nudge each axis under power (for geared axes "
                        "you can't move by hand, e.g. pan). Reads back present position. No ZMQ.")
    p.add_argument("--jog-step", type=int, default=6,
                   help="Ticks per tick added while an arrow is held in --jog (6 ~= 25 deg/s).")
    p.add_argument("--jog-max-lag", type=int, default=150,
                   help="Stall guard: stop advancing a goal if present pos lags it by this "
                        "many ticks (means the axis hit a stop). Keep small for geared axes.")
    return p.parse_args()


def rad_to_tick(angle_rad, gear, sign, home):
    """Convert a head-frame angle (rad) to a Dynamixel goal tick."""
    return int(round(home + sign * gear * angle_rad * TICKS_PER_RAD))


def _write1(packet, port, dxl_id, addr, value, what):
    r, err = packet.write1ByteTxRx(port, dxl_id, addr, value)
    if r != COMM_SUCCESS:
        raise RuntimeError(f"{what} on ID {dxl_id} failed: {packet.getTxRxResult(r)}")
    if err != 0:
        raise RuntimeError(f"{what} on ID {dxl_id} error: {packet.getRxPacketError(err)}")


def configure_servo(port, packet, current_id, current_baud, new_id, new_baud):
    """Write a new ID (and baud) to the ONE servo currently on the bus."""
    print("CONFIGURE -- make sure ONLY ONE servo is connected to the bus.\n"
          f"  targeting servo at id={current_id} baud={current_baud}")
    if not port.setBaudRate(current_baud):
        sys.exit(f"Failed to set baud {current_baud}")
    _, r, _ = packet.ping(port, current_id)
    if r != COMM_SUCCESS:
        sys.exit(f"No servo at id={current_id} baud={current_baud}. Run --scan to find it.")

    # ID and baud live in EEPROM -> torque must be off to change them.
    _write1(packet, port, current_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "torque disable")
    _write1(packet, port, current_id, ADDR_ID, new_id, "set ID")
    print(f"  ID {current_id} -> {new_id}")

    if new_baud != current_baud:
        if new_baud not in BAUD_ENUM:
            sys.exit(f"Unsupported baud {new_baud}; choose one of {sorted(BAUD_ENUM)}")
        # address by the NEW id (already changed), still at current wire baud
        _write1(packet, port, new_id, ADDR_BAUD_RATE, BAUD_ENUM[new_baud], "set baud")
        print(f"  baud {current_baud} -> {new_baud}")

    print("Done. Power-cycle the servo, then run --scan to confirm the new id/baud.")


def scan_bus(port, packet):
    """Broadcast-ping across common baud rates to locate every servo's ID + baud."""
    bauds = [2000000, 1000000, 57600, 115200, 3000000, 4000000, 9600, 500000, 200000]
    print("Scanning for Dynamixels across baud rates (this takes a few seconds)...")
    found = []
    for baud in bauds:
        if not port.setBaudRate(baud):
            continue
        data, comm = packet.broadcastPing(port)
        if comm == COMM_SUCCESS and data:
            for dxl_id in sorted(data):
                model = data[dxl_id][0]
                print(f"  FOUND  id={dxl_id:3d}  baud={baud}  model={model}")
                found.append((dxl_id, baud))
        else:
            print(f"  baud {baud}: none")
    if not found:
        print("\nNo servos found at ANY baud. That points to power or wiring, not config:\n"
              "  - Hub switch in the D2-LED-ON position? (servo ports actually energized)\n"
              "  - ~11-12V present at the servos? U2D2->Hub TTL cable seated?\n"
              "  - If power/wiring are definitely good, a servo may have been damaged.")
    else:
        print(f"\nRun the driver with the values above, e.g.:\n"
              f"  python scripts/onboard_neck_driver.py --calibrate "
              f"--baud {found[0][1]} --yaw-id {found[0][0]}"
              + (f" --pitch-id {found[1][0]}" if len(found) > 1 else ""))
    return found


def read_present_position(packet, port, dxl_id):
    """Return the servo's present position tick (0..4095 single-turn)."""
    val, r, err = packet.read4ByteTxRx(port, dxl_id, ADDR_PRESENT_POSITION)
    if r != COMM_SUCCESS or err != 0:
        return None
    if val >= 0x80000000:  # signed (extended/multi-turn)
        val -= 0x100000000
    return val


def calibrate_loop(packet, port, yaw_id, pitch_id):
    """Torque OFF; live-print present positions so you can read calibration by hand."""
    for dxl_id in (yaw_id, pitch_id):
        _, r, err = packet.ping(port, dxl_id)
        if r != COMM_SUCCESS:
            raise RuntimeError(f"Servo ID {dxl_id} not responding: {packet.getTxRxResult(r)}")
        _write1(packet, port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "torque disable")
    print(
        "\nCALIBRATE (torque OFF -- move the head by hand):\n"
        "  1. Hold head LEVEL + FORWARD -> read 'tick' = your --pan-home / --tilt-home.\n"
        "  2. Move each axis to its mechanical stops -> 'rad' at the stops = your limits.\n"
        "  3. Direction: turn an axis the way you call POSITIVE; if 'tick' rises use "
        "--*-sign 1, if it falls use -1.\n"
        "  ('rad' below is computed against home=2048; re-read after you fix home.)\n"
        "  Ctrl-C to quit.\n"
    )
    try:
        while True:
            yaw = read_present_position(packet, port, yaw_id)
            pitch = read_present_position(packet, port, pitch_id)
            def fmt(t):
                if t is None:
                    return "  READ-FAIL"
                return f"tick {t:5d}  ({(t - 2048) / TICKS_PER_RAD:+.3f} rad)"
            print(f"\r  yaw/ID{yaw_id}: {fmt(yaw)}    pitch/ID{pitch_id}: {fmt(pitch)}   ",
                  end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDone.")


def _write_goal(packet, port, dxl_id, tick):
    packet.write4ByteTxRx(port, dxl_id, ADDR_GOAL_POSITION, tick & 0xFFFFFFFF)


def jog_loop(packet, port, args, op_mode):
    """Torque ON; arrow keys nudge each axis so you can calibrate geared axes (pan)."""
    try:
        from pynput import keyboard
    except ImportError:
        sys.exit("--jog needs pynput:  pip install pynput")

    for dxl_id in (args.yaw_id, args.pitch_id):
        setup_servo(packet, port, dxl_id, op_mode)
    single_turn = (op_mode == OP_MODE_POSITION)

    yaw_goal = read_present_position(packet, port, args.yaw_id) or 2048
    pitch_goal = read_present_position(packet, port, args.pitch_id) or 2048

    held, stop = set(), {"q": False}

    def on_press(k):
        held.add(k)
        if k in (keyboard.Key.esc,) or getattr(k, "char", None) == "q":
            stop["q"] = True

    def on_release(k):
        held.discard(k)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print(
        "\nJOG (torque ON):  Left/Right = pan   Up/Down = tilt   q/Esc = quit\n"
        "  Move SLOWLY, stop at any resistance. Read 'pres' at:\n"
        "   - head level+forward -> your --pan-home / --tilt-home\n"
        "   - each mechanical stop -> the range for --pan-limit / --tilt-limit\n"
        "  ('rad' is vs tick 2048; recompute against your real home afterwards.)\n"
    )

    def step_axis(dxl_id, goal, dec_key, inc_key):
        pres = read_present_position(packet, port, dxl_id)
        if pres is None:
            return goal, pres
        # Only advance if the servo is keeping up (not stalled at a stop).
        if dec_key in held and (goal - pres) > -args.jog_max_lag:
            goal -= args.jog_step
        if inc_key in held and (goal - pres) < args.jog_max_lag:
            goal += args.jog_step
        if single_turn:
            goal = max(0, min(4095, goal))
        _write_goal(packet, port, dxl_id, goal)
        return goal, pres

    dt = 1.0 / 50.0
    try:
        while not stop["q"]:
            t0 = time.monotonic()
            yaw_goal, yaw_p = step_axis(args.yaw_id, yaw_goal,
                                        keyboard.Key.left, keyboard.Key.right)
            pitch_goal, pitch_p = step_axis(args.pitch_id, pitch_goal,
                                            keyboard.Key.down, keyboard.Key.up)

            def fmt(p):
                return "READ-FAIL" if p is None else f"{p:5d} ({(p-2048)/TICKS_PER_RAD:+.3f}rad)"
            print(f"\r  pan/ID{args.yaw_id} pres {fmt(yaw_p)}   "
                  f"tilt/ID{args.pitch_id} pres {fmt(pitch_p)}   ", end="", flush=True)

            sleep = dt - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        listener.stop()
        for dxl_id in (args.yaw_id, args.pitch_id):
            try:
                _write1(packet, port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "torque disable")
            except Exception:
                pass
        print("\nDone.")


_HW_ERROR_BITS = {0: "input-voltage", 2: "overheating", 3: "encoder",
                  4: "electrical-shock", 5: "overload"}


def _hw_error_names(hw):
    return ",".join(n for b, n in _HW_ERROR_BITS.items() if hw & (1 << b)) or f"0x{hw:02X}"


def clear_hardware_error(packet, port, dxl_id):
    """If the servo latched a hardware error (overload/voltage/overheat), report
    which one and reboot it to clear -- same effect as a manual power-cycle."""
    hw, r, _ = packet.read1ByteTxRx(port, dxl_id, ADDR_HARDWARE_ERROR)
    if r == COMM_SUCCESS and hw:
        print(f"  servo ID {dxl_id}: hardware error [{_hw_error_names(hw)}] -> rebooting to clear")
        packet.reboot(port, dxl_id)
        time.sleep(0.5)  # let it come back up before we talk to it again


def setup_servo(packet, port, dxl_id, op_mode):
    """Ping, clear any latched hardware error, set operating mode, enable torque."""
    _, r, err = packet.ping(port, dxl_id)
    if r != COMM_SUCCESS:
        raise RuntimeError(f"Servo ID {dxl_id} not responding: {packet.getTxRxResult(r)} "
                           f"(check wiring, baud, and the ID assignment)")
    clear_hardware_error(packet, port, dxl_id)
    _write1(packet, port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "torque disable")
    _write1(packet, port, dxl_id, ADDR_OPERATING_MODE, op_mode, "set operating mode")
    _write1(packet, port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE, "torque enable")
    print(f"  servo ID {dxl_id}: mode={op_mode} torque=ON")


def sync_write_goals(sync_writer, packet, pairs):
    """pairs: list of (dxl_id, goal_tick). Handles negative ticks (extended mode)."""
    sync_writer.clearParam()
    for dxl_id, goal in pairs:
        g = goal & 0xFFFFFFFF  # 4-byte little-endian, two's complement for <0
        param = [g & 0xFF, (g >> 8) & 0xFF, (g >> 16) & 0xFF, (g >> 24) & 0xFF]
        if not sync_writer.addParam(dxl_id, bytes(param)):
            raise RuntimeError(f"GroupSyncWrite.addParam failed for ID {dxl_id}")
    r = sync_writer.txPacket()
    if r != COMM_SUCCESS:
        # Non-fatal: log and keep going (a dropped bus frame shouldn't kill teleop).
        print(f"\n  [warn] sync write: {packet.getTxRxResult(r)}", flush=True)


def main():
    args = parse_args()
    op_mode = OP_MODE_EXTENDED if args.extended else OP_MODE_POSITION

    def goal_pairs(pan, tilt):
        pt = rad_to_tick(pan * args.pan_scale, args.pan_gear, args.pan_sign, args.pan_home)
        tt = rad_to_tick(tilt * args.tilt_scale, args.tilt_gear, args.tilt_sign, args.tilt_home)
        pt = max(args.pan_min, min(args.pan_max, pt))    # clamp to mechanical stops
        tt = max(args.tilt_min, min(args.tilt_max, tt))
        return [(args.yaw_id, pt), (args.pitch_id, tt)]

    # ── Scan / configure: open bus, do the one-shot task, then exit ──
    if args.scan or args.configure:
        port = PortHandler(args.device)
        packet = PacketHandler(PROTOCOL_VERSION)
        if not port.openPort():
            sys.exit(f"Failed to open {args.device} (plugged in? permissions?)")
        try:
            if args.configure:
                if args.new_id is None:
                    sys.exit("--configure requires --new-id")
                configure_servo(port, packet, args.current_id, args.current_baud,
                                args.new_id, args.new_baud)
            else:
                scan_bus(port, packet)
        finally:
            port.closePort()
        return

    # ── Calibration / jog: open bus, read positions, then exit (no ZMQ) ──
    if args.calibrate or args.jog:
        port = PortHandler(args.device)
        packet = PacketHandler(PROTOCOL_VERSION)
        if not port.openPort() or not port.setBaudRate(args.baud):
            sys.exit(f"Failed to open {args.device} @ {args.baud} baud.")
        try:
            if args.jog:
                jog_loop(packet, port, args, op_mode)
            else:
                calibrate_loop(packet, port, args.yaw_id, args.pitch_id)
        finally:
            port.closePort()
        return

    # ── Dynamixel bus up (skipped in --dry-run) ──
    port = packet = sync_writer = None
    if args.dry_run:
        print("DRY RUN: serial bus not opened, servos not moved -- printing ticks only.")
    else:
        port = PortHandler(args.device)
        packet = PacketHandler(PROTOCOL_VERSION)
        if not port.openPort():
            sys.exit(f"Failed to open {args.device} (is it plugged in? permissions?)")
        if not port.setBaudRate(args.baud):
            sys.exit(f"Failed to set baud {args.baud} on {args.device}")
        print(f"Opened {args.device} @ {args.baud} baud.")

        setup_servo(packet, port, args.yaw_id, op_mode)
        setup_servo(packet, port, args.pitch_id, op_mode)
        sync_writer = GroupSyncWrite(port, packet, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

        # Move to home (pan=0, tilt=0) before taking commands.
        sync_write_goals(sync_writer, packet, goal_pairs(0.0, 0.0))
        print("Homed to (pan=0, tilt=0).")

    pan, tilt = 0.0, 0.0

    # ── neck source: Redis (real, default) or ZMQ (sim relay/bridge on :5558) ──
    ctx = sub = rds = None
    if args.source == "redis":
        import json
        try:
            import redis
        except ImportError:
            sys.exit("--source redis needs the redis client:  pip install redis")
        rds = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
        print(f"Reading neck_target from Redis '{args.neck_key}' @ "
              f"{args.redis_host}:{args.redis_port}. Ctrl-C to quit.")
    else:
        try:
            import msgpack
            import msgpack_numpy as mnp
            import zmq
        except ImportError as e:
            sys.exit(f"--source zmq needs msgpack, msgpack-numpy and pyzmq ({e}). "
                     f"The real-robot path uses --source redis, which needs none of them.")
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.setsockopt(zmq.CONFLATE, True)
        sub.setsockopt(zmq.RCVHWM, 3)
        sub.connect(f"tcp://{args.host}:{args.port}")
        print(f"Subscribed to neck_target at tcp://{args.host}:{args.port}. Ctrl-C to quit.")

    dt = 1.0 / args.rate_hz
    last_rx = time.monotonic()
    last_t_action = None
    last_pub_change = time.monotonic()
    try:
        while True:
            t0 = time.monotonic()

            # Pull the freshest neck target from the active source. Mechanical clamp
            # happens in goal_pairs (tick space), so the asymmetric range is preserved.
            if args.source == "redis":
                # action_neck holds the latest [pan, tilt]; t_action (ms, stamped by
                # teleop) gives freshness so a dead publisher re-homes the neck rather
                # than freezing on a stale pose.
                try:
                    raw = rds.get(args.neck_key)
                    t_raw = rds.get("t_action")
                    # Liveness by CHANGE, not by absolute timestamp. t_action is
                    # stamped on the publisher's clock; comparing it to ours needs
                    # the two machines to agree, and the G1 Orin has no internet so
                    # NTP never syncs it. A 38 s skew made every target look ancient
                    # and the neck sat at home forever while everything else looked
                    # healthy. All we actually need to know is whether the publisher
                    # is still ticking -- so watch t_action advance against our own
                    # monotonic clock.
                    if t_raw is not None and t_raw != last_t_action:
                        last_t_action = t_raw
                        last_pub_change = time.monotonic()
                    stale = (time.monotonic() - last_pub_change) > args.dropout_hold_s
                    if raw is not None and not stale:
                        neck = json.loads(raw)
                        if neck is not None and len(neck) >= 2:
                            pan, tilt = float(neck[0]), float(neck[1])
                            last_rx = time.monotonic()
                    elif time.monotonic() - last_rx > args.dropout_hold_s:
                        pan, tilt = 0.0, 0.0
                except Exception as e:
                    print(f"\n  [warn] redis neck read: {e!r}", flush=True)
            else:
                # ZMQ: CONFLATE keeps only the latest goal dict.
                if sub.poll(timeout=0):
                    try:
                        goal = msgpack.unpackb(sub.recv(zmq.NOBLOCK),
                                               object_hook=mnp.decode, raw=False)
                        neck = goal.get("neck_target")
                        if neck is not None:
                            pan = float(neck[0])
                            tilt = float(neck[1])
                            last_rx = time.monotonic()
                    except Exception as e:
                        print(f"\n  [warn] bad goal msg: {e!r}", flush=True)
                elif time.monotonic() - last_rx > args.dropout_hold_s:
                    pan, tilt = 0.0, 0.0

            pairs = goal_pairs(pan, tilt)
            if not args.dry_run:
                sync_write_goals(sync_writer, packet, pairs)
            print(f"\r  pan={pan:+.3f} -> ID{pairs[0][0]} tick {pairs[0][1]:5d}   "
                  f"tilt={tilt:+.3f} -> ID{pairs[1][0]} tick {pairs[1][1]:5d}   ",
                  end="", flush=True)

            sleep = dt - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping.")
        if not args.dry_run:
            if not args.hold_torque_on_exit:
                for dxl_id in (args.yaw_id, args.pitch_id):
                    try:
                        _write1(packet, port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "torque disable")
                    except Exception:
                        pass
            port.closePort()
        if sub is not None:
            sub.close()
        if ctx is not None:
            ctx.term()


if __name__ == "__main__":
    main()
