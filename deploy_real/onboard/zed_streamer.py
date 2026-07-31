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
import struct
import sys
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
    "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
    "QUALITY": sl.DEPTH_MODE.QUALITY,
    "NEURAL": sl.DEPTH_MODE.NEURAL,
}


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
    p.add_argument("--preview", action="store_true", help="Show live RGB + depth windows")
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

    # ── ZMQ PUB (mirror run_isaac_sim_loop.py's camera socket: drop stale frames) ──
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.bind(f"tcp://*:{args.port}")
    print(f"Publishing binary RGB{'+depth' if want_depth else ''} on tcp://*:{args.port}. Ctrl-C to quit.")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    resize = args.width > 0 and args.height > 0
    frames, t_report = 0, time.monotonic()
    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            zed.retrieve_image(image, view)
            bgr = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)

            depth_u16 = None
            if want_depth:
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

            ok, jpeg = cv2.imencode(".jpg", bgr, encode_params)
            if not ok:
                continue
            jpeg_bytes = jpeg.tobytes()
            depth_bytes = depth_u16.tobytes() if depth_u16 is not None else b""

            header = struct.pack("iiii", w, h, len(jpeg_bytes), len(depth_bytes))
            sock.send(header + jpeg_bytes + depth_bytes, zmq.NOBLOCK)

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
        zed.close()
        sock.close()
        ctx.term()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
