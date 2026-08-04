"""Onboard ZED Mini streamer: RGB + aligned depth over ZMQ.

The ZED counterpart of realsense_streamer.py in this directory. Deployed to the
G1 Orin by deploy_to_robot.sh and started by start_zed.sh, same as the other
onboard sensors:

    :5555  realsense_streamer.py   D435i, chest-mounted, fixed
    :5556  THIS                    ZED Mini, neck-mounted -- moves with the head
    :5557  mid360_streamer.py      Livox MID-360 LiDAR

Captures the ZED Mini's left eye (RGB) plus the aligned metric depth and
publishes both in the SAME 16-byte-header format realsense_streamer.py uses, so
data_utils/vision_client.py reads it with no changes:

    [int32 width][int32 height][int32 rgb_jpeg_len][int32 depth_len]
    [JPEG BGR bytes][uint16 depth bytes, millimeters, row-major HxW]

  * RGB   -> JPEG-encoded left eye.
  * depth -> ZED SDK MEASURE.DEPTH, in millimeters, cast to uint16 and pixel-
    aligned to the RGB (invalid/NaN pixels = 0). Matches the RealSense
    convention and the sim's `distance_to_image_plane` (x1000 -> mm).
  * --no-depth sets depth_len = 0 (RGB only), same wire format.

Depth requires the ZED SDK (pyzed) + CUDA -- there is no UVC path for metric
depth, and it is NOT pip-installable, so it is absent from requirements.txt.
Install the JetPack-matching SDK from stereolabs.com on the Orin, then run its
get_python_api.py.

Runs on whichever machine the ZED is plugged into (the Orin for deployment, a
workstation for a bench test); consumers connect over the network.

Usage:
  python3 zed_streamer.py                       # RGB+depth, HD720
  python3 zed_streamer.py --preview             # + live windows
  python3 zed_streamer.py --no-depth            # RGB only
  python3 zed_streamer.py --width 424 --height 240 --fps 60
"""
import argparse
import os
import queue
import struct
import sys
import threading
import time

import cv2
import numpy as np
import zmq

try:
    import pyzed.sl as sl
except ImportError:
    sys.exit(
        "pyzed (ZED SDK Python API) is required for the ZED streamer.\n"
        "Install the ZED SDK from stereolabs.com (matching your JetPack/CUDA),\n"
        "then run its get_python_api.py. Depth needs the SDK -- there is no UVC path."
    )

RESOLUTIONS = {
    "VGA": sl.RESOLUTION.VGA,      # 672x376
    "HD720": sl.RESOLUTION.HD720,  # 1280x720 (matches the sim ZED camera)
    "HD1080": sl.RESOLUTION.HD1080,
    "HD2K": sl.RESOLUTION.HD2K,
}
DEPTH_MODES = {
    name: getattr(sl.DEPTH_MODE, name)
    for name in ("PERFORMANCE", "QUALITY", "NEURAL", "NEURAL_LIGHT", "NEURAL_PLUS")
    if hasattr(sl.DEPTH_MODE, name)
}

# SDK >= 4 exposes depth already quantised to uint16 millimetres, which is exactly
# our wire format -- so the float32 fetch plus nan_to_num/clip/astype is pure waste
# when it exists. None on older SDKs; the loop falls back to the float path.
DEPTH_MEASURE_U16 = getattr(sl.MEASURE, "DEPTH_U16_MM", None)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--port", type=int, default=5556, help="ZMQ PUB port (bridge SUB connects here)")
    p.add_argument("--resolution", choices=tuple(RESOLUTIONS), default="HD720", help="ZED capture resolution")
    p.add_argument("--fps", type=int, default=30, help="ZED capture FPS")
    p.add_argument("--jpeg-quality", type=int, default=85, help="RGB JPEG quality 0-100")
    p.add_argument("--width", type=int, default=0, help="Resize width before send (0 = native)")
    p.add_argument("--height", type=int, default=0, help="Resize height before send (0 = native)")
    p.add_argument("--no-depth", action="store_true", help="Stream RGB only (depth_len = 0)")
    p.add_argument("--depth-mode", choices=tuple(DEPTH_MODES), default="PERFORMANCE",
                   help="ZED depth computation mode (ignored with --no-depth)")
    p.add_argument("--view", choices=("left", "right"), default="left",
                   help="Which eye for the RGB ego-view (depth is always left-aligned)")
    p.add_argument("--stereo-port", type=int, default=0,
                   help="Also publish SIDE_BY_SIDE stereo (2x width, no depth) on this "
                        "port, for the PICO headset. The headset's ZEDMINI profile asks "
                        "for 2560x720 stereo; the mono :5556 stream gets stretched across "
                        "that rect and does not fill the FOV. 0 = disabled. Costs an extra "
                        "JPEG encode of a double-width frame, so watch the FPS counter.")
    p.add_argument("--preview", action="store_true", help="Show live RGB + depth windows")
    p.add_argument("--sync", action="store_true",
                   help="Encode and send on the capture thread (the old behaviour). "
                        "Costs ~half the frame rate -- see the note on the encoder thread.")
    p.add_argument("--settings-path", default=os.environ.get("ZED_SETTINGS_PATH", ""),
                   help="Directory holding the per-camera calibration (SN<serial>.conf). "
                        "Defaults to $ZED_SETTINGS_PATH. Use this when "
                        "/usr/local/zed/settings/ is not writable by the current user, "
                        "which makes open() fail with CALIBRATION FILE NOT AVAILABLE.")
    return p.parse_args()


def main():
    args = parse_args()
    want_depth = not args.no_depth

    # ── open the ZED ──
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = RESOLUTIONS[args.resolution]
    init.camera_fps = args.fps
    init.depth_mode = DEPTH_MODES[args.depth_mode] if want_depth else sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.MILLIMETER  # so MEASURE.DEPTH is in mm
    if args.settings_path:
        # The SDK caches the per-camera calibration (SN<serial>.conf) under
        # /usr/local/zed/settings/, which is typically root- or zed-group-owned.
        # Without write access the first-open download fails and the SDK reports
        # "CALIBRATION FILE NOT AVAILABLE" -- which reads like a broken camera but
        # is only a permission problem. Point it at a user-writable dir instead of
        # requiring sudo.
        init.optional_settings_path = args.settings_path
        print(f"ZED settings path: {args.settings_path}")
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        hint = "is the ZED Mini connected via USB 3.0?"
        if "CALIBRATION" in str(status).upper():
            hint = ("calibration file missing or its directory is not writable. "
                    "Fetch it to a writable dir and pass --settings-path (or set "
                    "ZED_SETTINGS_PATH):\n"
                    "  mkdir -p ~/.zed_settings && curl -sL -o ~/.zed_settings/SN<serial>.conf "
                    "'https://calib.stereolabs.com/?SN=<serial>'")
        sys.exit(f"Failed to open ZED: {status} ({hint})")
    print(f"ZED open: {args.resolution} @ {args.fps} FPS, "
          f"{'RGB+depth' if want_depth else 'RGB only'}, {args.view} eye.")

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth_mat = sl.Mat()
    view = sl.VIEW.LEFT if args.view == "left" else sl.VIEW.RIGHT

    # Optional stereo side-car for the headset. Kept as a SEPARATE socket rather
    # than replacing :5556, because the recorder and the GR00T bridge both expect
    # mono + aligned depth -- a double-width frame would break their depth
    # alignment. Same capture, two encodings.
    stereo_mat = sl.Mat() if args.stereo_port else None
    stereo_view = getattr(sl.VIEW, "SIDE_BY_SIDE", None)
    if args.stereo_port and stereo_view is None:
        sys.exit("This ZED SDK has no VIEW.SIDE_BY_SIDE; --stereo-port unavailable.")

    # ── ZMQ PUB (mirror run_isaac_sim_loop.py's camera socket: drop stale frames) ──
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.bind(f"tcp://*:{args.port}")
    print(f"Publishing binary RGB{'+depth' if want_depth else ''} on tcp://*:{args.port}. Ctrl-C to quit.")

    stereo_sock = None
    if args.stereo_port:
        stereo_sock = ctx.socket(zmq.PUB)
        stereo_sock.setsockopt(zmq.SNDHWM, 1)
        stereo_sock.bind(f"tcp://*:{args.stereo_port}")
        print(f"Publishing SIDE_BY_SIDE stereo (no depth) on tcp://*:{args.stereo_port} "
              f"-- point start_zed_headset.sh at this port.")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    resize = args.width > 0 and args.height > 0

    # ── encoder thread ──
    # grab() blocks until the camera's next frame (33 ms at 30 FPS) and the SDK
    # computes depth on the GPU inside that window, so the capture thread is idle
    # for most of it. Encoding and sending inline burned that idle time on the
    # critical path and halved the rate: measured on the Orin, HD720 + NEURAL was
    # 15 fps inline vs 30 fps with this thread -- at FULL depth quality. The depth
    # inference was never the bottleneck (grab() alone sustains 30 fps in every
    # depth mode); serialising JPEG + send behind it was.
    #
    # maxsize=1 with drop-on-full: a slow consumer must never make the camera lag,
    # and a dropped frame is better than a stale one. Matches SNDHWM=1 above.
    frame_q = queue.Queue(maxsize=1)
    _STOP = object()

    def encoder():
        while True:
            item = frame_q.get()
            if item is _STOP:
                return
            bgr_f, depth_bytes_f = item
            ok, jpeg = cv2.imencode(".jpg", bgr_f, encode_params)
            if not ok:
                continue
            jpeg_bytes = jpeg.tobytes()
            hh, ww = bgr_f.shape[:2]
            header = struct.pack("iiii", ww, hh, len(jpeg_bytes), len(depth_bytes_f))
            try:
                sock.send(header + jpeg_bytes + depth_bytes_f, zmq.NOBLOCK)
            except zmq.ZMQError:
                pass

    # The stereo frame is double-width, so encoding it costs roughly as much as
    # the mono one. Sharing a single thread made the two serialise and dragged
    # the mono stream -- which the recorder and bridge read -- from 30 to 19 fps.
    # Its own thread and queue keeps them independent: cv2.imencode releases the
    # GIL, so on the Orin's 8 cores they genuinely run in parallel, and a stall
    # on either socket can no longer slow the other.
    stereo_q = queue.Queue(maxsize=1)

    def stereo_encoder():
        while True:
            item = stereo_q.get()
            if item is _STOP:
                return
            ok, sj = cv2.imencode(".jpg", item, encode_params)
            if not ok:
                continue
            sb = sj.tobytes()
            sh, sw = item.shape[:2]
            try:
                stereo_sock.send(struct.pack("iiii", sw, sh, len(sb), 0) + sb, zmq.NOBLOCK)
            except zmq.ZMQError:
                pass

    enc_thread = None
    stereo_thread = None
    if not args.sync:
        enc_thread = threading.Thread(target=encoder, daemon=True)
        enc_thread.start()
        if stereo_sock is not None:
            stereo_thread = threading.Thread(target=stereo_encoder, daemon=True)
            stereo_thread.start()

    frames, t_report = 0, time.monotonic()
    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            zed.retrieve_image(image, view)
            bgr = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)

            stereo_bgr = None
            if stereo_sock is not None:
                # Deriving the mono view from this buffer's left half instead of a
                # separate VIEW.LEFT retrieve was tried and MEASURED NO FASTER
                # (23.6 vs 25.6 fps) -- the redundant GPU->CPU transfer was not the
                # bottleneck it looked like. Kept separate: simpler, and the mono
                # stream is what the recorder and bridge depend on.
                zed.retrieve_image(stereo_mat, stereo_view)
                stereo_bgr = cv2.cvtColor(stereo_mat.get_data(), cv2.COLOR_BGRA2BGR)

            depth_u16 = None
            if want_depth:
                if DEPTH_MEASURE_U16 is not None:
                    # Native uint16 mm straight from the SDK: 1.4 ms vs 5.1 ms for
                    # MEASURE.DEPTH + nan_to_num/clip/astype, which walks the whole
                    # 1280x720 float32 buffer three times. copy() because get_data()
                    # is a view onto the Mat, which the next retrieve overwrites --
                    # and the encoder thread reads it after we have moved on.
                    zed.retrieve_measure(depth_mat, DEPTH_MEASURE_U16)
                    depth_u16 = depth_mat.get_data().copy()
                else:
                    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)  # float32 mm, NaN=invalid
                    depth_f = depth_mat.get_data()
                    depth_u16 = np.nan_to_num(depth_f, nan=0.0, posinf=0.0, neginf=0.0)
                    depth_u16 = np.clip(depth_u16, 0, 65535).astype(np.uint16)

            if resize:
                bgr = cv2.resize(bgr, (args.width, args.height), interpolation=cv2.INTER_AREA)
                if depth_u16 is not None:
                    # nearest so depth values aren't blended across edges
                    depth_u16 = cv2.resize(depth_u16, (args.width, args.height),
                                           interpolation=cv2.INTER_NEAREST)
            h, w = bgr.shape[:2]

            depth_bytes = depth_u16.tobytes() if depth_u16 is not None else b""

            if enc_thread is not None:
                try:
                    frame_q.put_nowait((bgr, depth_bytes))
                except queue.Full:
                    pass  # encoder still busy -- drop this frame, never stall capture
                if stereo_thread is not None and stereo_bgr is not None:
                    try:
                        stereo_q.put_nowait(stereo_bgr)
                    except queue.Full:
                        pass  # headset lagging must never slow the mono stream
            else:
                ok, jpeg = cv2.imencode(".jpg", bgr, encode_params)
                if not ok:
                    continue
                jpeg_bytes = jpeg.tobytes()
                header = struct.pack("iiii", w, h, len(jpeg_bytes), len(depth_bytes))
                sock.send(header + jpeg_bytes + depth_bytes, zmq.NOBLOCK)
                if stereo_sock is not None and stereo_bgr is not None:
                    ok2, sj = cv2.imencode(".jpg", stereo_bgr, encode_params)
                    if ok2:
                        sb = sj.tobytes()
                        sh, sw = stereo_bgr.shape[:2]
                        stereo_sock.send(struct.pack("iiii", sw, sh, len(sb), 0) + sb,
                                         zmq.NOBLOCK)

            if args.preview:
                cv2.imshow("ZED RGB", bgr)
                if depth_u16 is not None:
                    dcol = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_u16, alpha=255.0 / 5000.0),  # 0-5 m -> 0-255
                        cv2.COLORMAP_TURBO)
                    cv2.imshow("ZED depth (mm)", dcol)
                cv2.waitKey(1)

            frames += 1
            now = time.monotonic()
            if now - t_report >= 2.0:
                print(f"\r  Frames: {frames}  FPS: {frames / (now - t_report):5.1f}  "
                      f"{w}x{h}   ", end="", flush=True)
                frames, t_report = 0, now
    except KeyboardInterrupt:
        pass
    except zmq.ZMQError:
        pass
    finally:
        print("\nStopping.")
        if enc_thread is not None:
            try:
                frame_q.put_nowait(_STOP)
            except queue.Full:
                frame_q.get_nowait()      # make room; the pending frame is moot now
                frame_q.put_nowait(_STOP)
            enc_thread.join(timeout=2.0)
        if stereo_thread is not None:
            try:
                stereo_q.put_nowait(_STOP)
            except queue.Full:
                stereo_q.get_nowait()
                stereo_q.put_nowait(_STOP)
            stereo_thread.join(timeout=2.0)
        zed.close()
        sock.close()
        ctx.term()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
