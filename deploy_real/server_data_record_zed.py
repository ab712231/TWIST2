#!/usr/bin/env python3

"""
Data collection script for the neck-mounted ZED Mini (RGB + aligned depth).

Collects RGB + metric depth from onboard/zed_streamer.py (via ZMQ), plus
body/hand state and action from Redis, and writes episodes to disk.

This REPLACES the ZED path in server_data_record.py, which was written for the
old OrinVideoSender: that one subscribes to :5555 expecting a side-by-side
stereo frame (360x1280) and saves RGB only. zed_streamer.py publishes a single
rectified left eye plus aligned uint16 depth in the same 16-byte-header wire
format the D435i uses, so this recorder is the D435i one with ZED geometry.

Why the ZED and not the D435i for imitation data: the D435i is fixed in the
chest, while the ZED Mini rides the neck and moves with the operator's head, so
its view is the one that actually corresponds to what the teleoperator saw.

Differences from server_data_record_d435.py:
  - 1280x720 single eye (not 424x240)
  - :5556 (not :5555)
  - 30 Hz camera, so --camera_fps defaults to 30; the subsample cadence is
    derived from it instead of being hardcoded to 60
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import redis
from datetime import datetime
from multiprocessing import shared_memory
import threading

from data_utils.episode_writer import EpisodeWriter
from data_utils.vision_client import VisionClient
from rich import print
from robot_control.speaker import Speaker


def main(args):
    # ---- Redis connection ----
    try:
        redis_pool = redis.ConnectionPool(
            host="localhost",
            port=6379,
            db=0,
            max_connections=10,
            retry_on_timeout=True,
            socket_timeout=0.1,
            socket_connect_timeout=0.1,
        )
        redis_client = redis.Redis(connection_pool=redis_pool)
        redis_pipeline = redis_client.pipeline()
        redis_client.ping()
        print(f"Connected to Redis at localhost:6379, DB=0 with connection pool")
    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        return

    # ---- Shared memory for RGB (1280x720x3) ----
    image_shape = (args.height, args.width, 3)
    image_shared_memory = shared_memory.SharedMemory(
        create=True, size=int(np.prod(image_shape) * np.uint8().itemsize)
    )
    image_array = np.ndarray(image_shape, dtype=np.uint8, buffer=image_shared_memory.buf)

    # ---- Shared memory for depth (1280x720, float32) ----
    depth_shape = (args.height, args.width)
    depth_shared_memory = shared_memory.SharedMemory(
        create=True, size=int(np.prod(depth_shape) * np.float32().itemsize)
    )
    depth_array = np.ndarray(depth_shape, dtype=np.float32, buffer=depth_shared_memory.buf)

    # ---- Vision client ----
    try:
        cv2.namedWindow("__test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__test__")
        image_show = True
    except cv2.error:
        image_show = False
        print("[Warning] OpenCV GUI not available (headless build). Disabling image preview.")
    vision_client = VisionClient(
        server_address=args.robot_ip,
        port=args.camera_port,
        img_shape=image_shape,
        img_shm_name=image_shared_memory.name,
        depth_shape=depth_shape,
        depth_shm_name=depth_shared_memory.name,
        image_show=False,
        depth_show=False,
        unit_test=True,
    )
    vision_thread = threading.Thread(target=vision_client.receive_process, daemon=True)
    vision_thread.start()

    # ---- Episode writer ----
    recording = False
    save_data_keys = ["rgb", "depth"]
    task_dir = os.path.join(args.data_folder, args.task_name)
    recorder = EpisodeWriter(
        task_dir=task_dir,
        frequency=args.frequency,
        image_shape=image_shape,
        data_keys=save_data_keys,
    )
    recorder.text_desc(goal=args.goal, desc=args.desc, steps=args.steps)

    control_dt = 1.0 / args.frequency
    step_count = 0
    frame_counter = 0  # for sub-sampling
    running = True

    # Sub-sampling: how many camera frames to skip between recorded frames.
    # e.g. camera at 30fps, --frequency 15 → record every 2nd frame. Derived from
    # --camera_fps rather than hardcoded, because the ZED runs 30 and the D435i 60.
    subsample_interval = max(1, round(float(args.camera_fps) / args.frequency))

    print(f"Recorded control frequency: {args.frequency} Hz (subsample every {subsample_interval} frames)")

    speaker = Speaker()
    prev_button_pressed = False
    prev_right_axis_click_pressed = False

    # Liveness state for the controller feed (see the loop for why this exists).
    CONTROLLER_STALE_S = 2.0
    last_raw_controller = None
    last_controller_change = time.monotonic()
    controller_stale = False
    warned_missing = False

    try:
        while running:
            start_time = time.time()

            # ---- Controller input ----
            # Redis keys never expire (ttl -1), so a dead teleop leaves its LAST
            # controller_data lying there forever and this loop happily reads it as
            # if it were live. The buttons then do nothing -- the toggle below is
            # rising-edge, so a value frozen either way never fires -- and it looks
            # like a broken camera rather than a missing publisher. Worse, a session
            # that ended with axis_click True would quit us instantly on startup.
            # So gate on the payload actually CHANGING, on the monotonic clock (the
            # robot's wall clock drifts seconds from the workstation's).
            raw_controller = redis_client.get("controller_data")
            if raw_controller is None:
                if not warned_missing:
                    print("[controller] no controller_data in Redis -- is teleop.sh running?", flush=True)
                    warned_missing = True
                time.sleep(0.1)
                continue
            warned_missing = False
            if raw_controller != last_raw_controller:
                last_raw_controller = raw_controller
                last_controller_change = time.monotonic()
                if controller_stale:
                    controller_stale = False
                    print("[controller] feed live again.", flush=True)
            elif not controller_stale and time.monotonic() - last_controller_change > CONTROLLER_STALE_S:
                controller_stale = True
                # Distinguish the two causes, because they need opposite fixes and
                # look identical from here. A live PICO stamps get_time_stamp_ns()
                # into every frame, so timestamp == 0 means teleop IS publishing but
                # the headset is not connected -- restarting teleop would not help.
                try:
                    ts = json.loads(raw_controller).get("timestamp", 0)
                except Exception:
                    ts = 0
                if ts:
                    why = "teleop.sh is not publishing -- is it still running?"
                else:
                    why = ("PICO not connected (timestamp=0) -- turn on the headset and "
                           "open the XRoboToolkit app. teleop.sh itself is fine.")
                print(f"[controller] FROZEN for {CONTROLLER_STALE_S:.0f}s -- buttons will not "
                      f"respond. {why}", flush=True)

            controller_data = json.loads(raw_controller)
            button_pressed = controller_data["LeftController"]["key_two"]

            quit_key = controller_data["LeftController"]["axis_click"]
            if quit_key:
                running = False
                speaker.speak("Recording stopped.")
                print("\nQuitting...")
                break

            right_axis_click = controller_data["RightController"]["axis_click"]

            # Rising-edge toggle
            if button_pressed and not prev_button_pressed:
                print("button pressed")
                recording = not recording
                if recording:
                    speaker.speak("episode recording started.")
                    if not recorder.create_episode():
                        recording = False
                    step_count = 0
                    frame_counter = 0
                    print("episode recording started...")
                else:
                    recorder.save_episode(label="successful")
                    speaker.speak("episode saved as successful.")

            # Right axis_click: save as unsuccessful
            if right_axis_click and not prev_right_axis_click_pressed and recording:
                recorder.save_episode(label="unsuccessful")
                recording = False
                speaker.speak("episode saved as unsuccessful.")

            prev_button_pressed = button_pressed
            prev_right_axis_click_pressed = right_axis_click

            if recording:
                frame_counter += 1

                # Sub-sample: only record on the correct cadence
                if frame_counter % subsample_interval != 0:
                    elapsed = time.time() - start_time
                    if elapsed < control_dt:
                        time.sleep(control_dt - elapsed)
                    continue

                data_dict = {"idx": step_count}

                # ---- Vision data ----
                data_dict["rgb"] = image_array.copy()
                data_dict["t_img"] = int(time.time() * 1000)

                # Depth: read shared memory and convert float32 → uint16 (mm)
                raw_depth = depth_array.copy()
                if np.any(raw_depth > 0):
                    data_dict["depth"] = raw_depth.astype(np.uint16)
                else:
                    data_dict["depth"] = None

                # ---- Redis state & action ----
                redis_keys = [
                    "state_body_unitree_g1_with_hands",
                    "state_hand_left_unitree_g1_with_hands",
                    "state_hand_right_unitree_g1_with_hands",
                    "state_neck_unitree_g1_with_hands",
                    "t_state",
                    "action_body_unitree_g1_with_hands",
                    "action_hand_left_unitree_g1_with_hands",
                    "action_hand_right_unitree_g1_with_hands",
                    "action_neck_unitree_g1_with_hands",
                    "t_action",
                    "action_low_level_unitree_g1_with_hands",
                    "force_hand_left_unitree_g1_with_hands",
                    "force_hand_right_unitree_g1_with_hands",
                    "tactile_hand_left_unitree_g1_with_hands",
                    "tactile_hand_right_unitree_g1_with_hands",
                ]

                data_dict_keys = [
                    "state_body",
                    "state_hand_left",
                    "state_hand_right",
                    "state_neck",
                    "t_state",
                    "action_body",
                    "action_hand_left",
                    "action_hand_right",
                    "action_neck",
                    "t_action",
                    "action_low_level",
                    "force_hand_left",
                    "force_hand_right",
                    "tactile_hand_left",
                    "tactile_hand_right",
                ]

                try:
                    for key in redis_keys:
                        redis_pipeline.get(key)
                    redis_results = redis_pipeline.execute()

                    for i, (result, dict_key) in enumerate(zip(redis_results, data_dict_keys)):
                        if result is not None:
                            try:
                                data_dict[dict_key] = json.loads(result)
                            except json.JSONDecodeError:
                                print(f"Warning: Failed to decode JSON for key {redis_keys[i]}")
                                data_dict[dict_key] = None
                        else:
                            print(f"Warning: No data found for key {redis_keys[i]}")
                            data_dict[dict_key] = None
                except Exception as e:
                    print(f"Error in Redis pipeline operation: {e}")
                    continue

                recorder.add_item(data_dict)

                if image_show:
                    if image_array is not None and image_array.size > 0:
                        window_name = "ZED Mini - Press controller button to start/stop recording"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, image_array.shape[1], image_array.shape[0])
                        cv2.moveWindow(window_name, 50, 50)
                        cv2.imshow(window_name, image_array)
                        cv2.waitKey(1)

                step_count += 1
                elapsed = time.time() - start_time
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)
            else:
                if image_show:
                    if image_array is not None and image_array.size > 0:
                        window_name = "ZED Mini - Press controller button to start/stop recording"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, image_array.shape[1], image_array.shape[0])
                        cv2.moveWindow(window_name, 50, 50)
                        cv2.imshow(window_name, image_array)
                        cv2.waitKey(1)
                else:
                    time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, exiting...")
        running = False
    finally:
        print(f"\nDone! Recorded {recorder.episode_id + 1} episodes to {task_dir}")

        image_shared_memory.unlink()
        image_shared_memory.close()
        depth_shared_memory.unlink()
        depth_shared_memory.close()
        recorder.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        print("Exiting the recording...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED Mini data recording with RGB + depth.")
    cur_time = datetime.now().strftime("%Y%m%d_%H%M")

    parser.add_argument("--data_folder", default="twist2_demonstration", help="Data folder")
    parser.add_argument("--task_name", default=f"{cur_time}", help="Task name")
    parser.add_argument("--frequency", default=30, type=int, help="Recording frequency in Hz")
    parser.add_argument("--robot", default="unitree_g1", choices=["unitree_g1"], help="Robot name")
    parser.add_argument("--robot_ip", default="192.168.123.164", help="Robot / Orin IP (where zed_streamer.py runs)")
    parser.add_argument("--camera_port", default=5556, type=int, help="ZMQ port for the ZED streamer")
    parser.add_argument("--camera_fps", default=30, type=int,
                        help="Capture rate of the streamer; sets the record subsample cadence")
    parser.add_argument("--width", default=1280, type=int, help="Image width")
    parser.add_argument("--height", default=720, type=int, help="Image height")

    # Task description metadata
    parser.add_argument("--goal", default="pick up the red cup", help="Task goal description")
    parser.add_argument("--desc", default="A humanoid robot picks up a red cup from the table.", help="Task description")
    parser.add_argument("--steps", default="step1: approach table. step2: grasp cup. step3: lift cup.", help="Task steps")

    args = parser.parse_args()
    main(args)
