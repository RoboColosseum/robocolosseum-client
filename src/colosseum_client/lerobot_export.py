"""Export one recorded trial as a standalone LeRobot v3 video dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from .roboeval_export import source_extension, stage_extension, validate_uploaded_extension, digest as file_digest
from .navigation import ACTION_NAMES, GOAL_GPS_KEY, NAV_STATE_KEY, NAV_STATE_NAMES, ROVER_TYPE


# LeRobot's so101_follower feature names, so exports align with training datasets.
SO101_NAMES = ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos")


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {**{name: function(values, axis=0).tolist() for name, function in
               [("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std)]},
            "count": [len(values)]}


def _clear_source_images(path, cameras, rows):
    """Only remove this recording's frames, after the export is committed."""
    for camera in cameras:
        for row in rows:
            (path / camera / f"{row['frame']:06d}.png").unlink(missing_ok=True)


from .dataset_guard import locked_export


@locked_export
def export_trial(path: Path, cameras: list[str], *, fps: int, robot_type: str,
                 task: str, metadata: dict, continuous_training_fps: int = 100,
                 export_destination: Path | None = None, omit_auxiliary_columns: bool = False, training_delivery: bool = False) -> Path:
    """Write atomically; retries reuse only a matching, complete export.

    Dataset timestamps index control steps at nominal FPS. The original irregular
    capture clock is preserved separately, without resampling observations/actions.
    Source PNGs are removed after export, without waiting for upload. The final
    observation with no action retains only its numeric source record.
    """
    path = Path(path)
    if fps <= 0 or not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("Positive FPS and unique cameras are required")
    if any(camera not in {"head_image", "left_image", "right_image", "front_image", "rear_image"} for camera in cameras):
        raise ValueError("Unknown camera")
    status = json.loads((path / "recording-status.json").read_text())
    if not status["complete"]:
        raise ValueError("Cannot export an incomplete recording")
    raw = (path / "frames.jsonl").read_bytes()
    all_rows = [json.loads(line) for line in raw.splitlines()]
    if status["frames"] != len(all_rows):
        raise ValueError("Recording frame count differs from its status")
    rows = [row for row in all_rows if row["action"] is not None]
    if not rows or rows != all_rows[:len(rows)] or len(all_rows) - len(rows) > 1:
        raise ValueError("Expected action frames followed by at most one final observation")
    if [row["frame"] for row in all_rows] != list(range(len(all_rows))):
        raise ValueError("Recording frame indices are not contiguous")
    recording_context = path / 'recording-context.json'
    recording = json.loads(recording_context.read_text()) if recording_context.exists() else {}
    continuous = None
    if status.get('image_storage') == 'continuous_source' and not recording.get('roboeval_state_hz'):
        raise ValueError('Source-video recording requires continuous export; PNG fallback is unavailable')
    if recording.get('roboeval_state_hz'):
        from .continuous_recording import build_continuous
        if robot_type != 'yam' or metadata.get('action_space') != 'joint_position':
            raise ValueError('Continuous export currently requires YAM joint policy targets')
        continuous = build_continuous(path, rows, recording, cameras, training_fps=continuous_training_fps)
        rows, fps = continuous['rows'], continuous_training_fps
    for row in rows:
        for key in ("joints", "gripper", "action"):
            value = np.asarray(row.get(key), dtype=np.float32)
            # A mobile base records empty joints and gripper; arms must have both.
            required = key == "action" or robot_type != ROVER_TYPE
            if value.ndim != 1 or (required and not value.size) or not np.isfinite(value).all():
                raise ValueError(f"Missing or invalid {key}; older recordings may lack full state")
    cartesian = [np.asarray(row.get("cartesian_position"), dtype=np.float32) for row in rows]
    if any(v.ndim != 1 or not np.isfinite(v).all() for v in cartesian):
        raise ValueError("Missing or invalid cartesian_position")
    if len({v.shape for v in cartesian}) != 1:
        raise ValueError("Inconsistent cartesian_position shape")
    vectors = {
        "observation.state": [row["joints"] + row["gripper"] for row in rows],
        "action": [row["action"] for row in rows],
    }
    if training_delivery:
        if continuous is None or fps != 30 or not omit_auxiliary_columns:
            raise ValueError('Training delivery requires continuous clean 30 Hz data')
        from .training_delivery import physical_vectors
        raw_meta = path / 'roboeval/meta/roboeval'
        vectors.update(physical_vectors(rows, pq.read_table(raw_meta/'states.parquet').to_pylist(),
                                        pq.read_table(raw_meta/'actions.parquet').to_pylist()))
    efforts = [row.get('gripper_effort_nm') for row in rows]
    if any(value is not None for value in efforts):
        if any(value is None for value in efforts) or not np.isfinite(efforts).all():
            raise ValueError('Incomplete or invalid gripper effort recording')
        vectors['observation.gripper_effort_nm'] = efforts
    if cartesian[0].size:
        vectors["observation.cartesian_position"] = cartesian
    policy_actions = [row.get('policy_action') for row in rows]
    if any(value is not None for value in policy_actions):
        if any(value is None for value in policy_actions):
            raise ValueError('Incomplete EE policy action recording')
        targets = np.asarray(policy_actions, dtype=np.float32)
        if targets.shape != (len(rows), 14) or not np.isfinite(targets).all():
            raise ValueError('YAM EE policy actions must be finite 14-D targets')
        vectors['action.eef_target'] = targets
    if robot_type == "yam":
        if any(len(row['joints']) != 12 or len(row['gripper']) != 2 or len(row['action']) != 14 for row in rows):
            raise ValueError("YAM recording requires 12 joints, 2 grippers and 14-D actions")
        vectors["observation.state"] = [row['joints'][:6] + row['gripper'][:1]
                                        + row['joints'][6:] + row['gripper'][1:] for row in rows]
    if robot_type == "so101" and any(
            len(row['joints']) != 5 or len(row['gripper']) != 1 or len(row['action']) != 6 for row in rows):
        raise ValueError("SO101 recording requires 5 joints, 1 gripper and 6-D actions")
    if robot_type == ROVER_TYPE:
        navigation = [(row.get("extra_state") or {}).get(NAV_STATE_KEY) for row in rows]
        if (any(row["joints"] or row["gripper"] or len(row["action"]) != 2 for row in rows)
                or any(value is None or len(value) != len(NAV_STATE_NAMES) for value in navigation)
                or not np.isfinite(np.asarray(navigation, dtype=np.float64)).all()):
            raise ValueError("Earth Rover recording requires no joints, 2-D actions and a complete navigation_state")
        vectors["observation.state"] = navigation
        goals = [(row.get("extra_state") or {}).get(GOAL_GPS_KEY) for row in rows]
        if any(goal is not None for goal in goals):
            if any(goal is None or len(goal) != 2 for goal in goals):
                raise ValueError("Incomplete goal_gps recording")
            vectors["observation.goal_gps"] = goals
    # GPS needs float64: float32 latitude/longitude is only metre-level.
    precise = {"observation.state", "observation.goal_gps"} if robot_type == ROVER_TYPE else set()
    vectors = {key: np.asarray(values, dtype=np.float64 if key in precise else np.float32)
               for key, values in vectors.items()}
    capture = np.asarray([row["seconds"] for row in rows], dtype=np.float64)
    if not np.isfinite(capture).all() or np.any(capture < 0) or np.any(np.diff(capture) <= 0):
        raise ValueError("Capture times must be finite, nonnegative and strictly increasing")
    policy_path = path / "policy.json"
    context = {**metadata, "robot_type": robot_type, "task": task, "fps": fps,
               "cameras": cameras, "policy": json.loads(policy_path.read_text()) if policy_path.exists() else None,
               "timestamp_semantics": "frame_index / nominal control FPS; no resampling",
               "capture_timestamp_semantics": "original seconds since recorder start",
               "action_semantics": ("driver-accepted joint command after EEF IK"
                                    if metadata.get("action_recording") == "accepted_joint_commands_after_eef_ik"
                                    else "selected command; not measured execution")}
    if training_delivery:
        from .training_delivery import PROFILE
        context['delivery_profile'] = PROFILE
    if omit_auxiliary_columns:
        if continuous is None:
            raise ValueError('Clean training columns require continuous source evidence')
        context['training_auxiliary_columns'] = 'omitted'
    if 'observation.gripper_effort_nm' in vectors:
        context['gripper_effort'] = dict(unit='N.m', order=['left', 'right'],
            source='signed motor 7 torque feedback', is_fingertip_force=False)
    if continuous is not None:
        context.update(continuous_recording=continuous['metadata'],
                       timestamp_semantics=f'{fps} Hz real elapsed grid after first successful action',
                       capture_timestamp_semantics='grid seconds; actual source samples retained separately')
    recording_context = path / 'recording-context.json'
    shared_required = (recording_context.exists() and
        json.loads(recording_context.read_text()).get('roboeval_shared_required') is True)
    shared = None
    if shared_required or metadata.get('roboeval_outcome') is not None:
        if metadata.get('roboeval_outcome') is None:
            raise ValueError('Confirmed comparison feedback is required before final export')
        from .roboeval_shared import shared_payload
        shared = shared_payload(path, metadata['roboeval_outcome'])
        context['roboeval_shared_source_sha256'] = shared['signature']
    if robot_type == "yam":
        context["state_action_order"] = "left_joint_1..6,left_gripper,right_joint_1..6,right_gripper"
        context["state_action_units"] = "radians; grippers normalized 0 closed, 1 open"
        context["cartesian_state_available"] = training_delivery or bool(cartesian[0].size)
    if robot_type == "so101":
        context["state_action_order"] = ",".join(SO101_NAMES)
        context["state_action_units"] = "degrees; gripper opening 0 closed, 100 open"
        context["cartesian_state_available"] = bool(cartesian[0].size)
    if robot_type == ROVER_TYPE:
        context["state_action_order"] = ",".join(NAV_STATE_NAMES) + " | action: " + ",".join(ACTION_NAMES)
        context["state_action_units"] = ("degrees (latitude, longitude, heading), m/s, percent, rpm; "
                                         "actions normalized [-1, 1], +angular turns left")
        context["cartesian_state_available"] = False
    extension = source_extension(path)
    if extension is not None:
        for key in ('assignment_id', 'run_id', 'policy'):
            if key in context and extension['identity'].get(key) != context[key]:
                raise ValueError('RoboEval source identity differs from this recording/context')
        context['roboeval_source_sha256'] = extension['signature']
    signature = hashlib.sha256(raw + json.dumps(context, sort_keys=True).encode()).hexdigest()
    destination = Path(export_destination) if export_destination is not None else path / "lerobot"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        marker = json.loads((destination / "meta/colosseum.json").read_text())
        if marker.get("source_sha256") != signature:
            raise ValueError("Existing LeRobot export differs from this recording/context")
        for relative, digest in marker["files"].items():
            if hashlib.sha256((destination / relative).read_bytes()).hexdigest() != digest:
                raise ValueError("Existing LeRobot export is incomplete or modified")
        validate_uploaded_extension(destination)
        if export_destination is None:
            _clear_source_images(path, cameras, all_rows)
        return destination
    staging = Path(tempfile.mkdtemp(prefix=".lerobot-", dir=destination.parent))
    compact_staging = None
    try:
        count = len(rows)
        features = {}
        columns = {}
        stats = {}
        joint_names = [f"joint_{i}" for i in range(len(rows[0]["joints"]))]
        gripper_names = [f"gripper_{i}" for i in range(len(rows[0]["gripper"]))]
        for key, array in vectors.items():
            names = joint_names + gripper_names if key == "observation.state" else [
                f"{key.rsplit('.', 1)[-1]}_{i}" for i in range(array.shape[1])]
            if robot_type == "yam" and key in {"observation.state", "action"}:
                names = [name for side in ('left', 'right')
                         for name in [*[f'{side}_joint_{i}' for i in range(1, 7)], f'{side}_gripper']]
            if robot_type == 'yam' and key == 'action.eef_target':
                names = [f'{side}_{axis}' for side in ('left', 'right')
                         for axis in ('x', 'y', 'z', 'yaw', 'pitch', 'roll', 'gripper')]
            if robot_type == "so101" and key in {"observation.state", "action"}:
                names = list(SO101_NAMES)
            if robot_type == ROVER_TYPE:
                names = {"observation.state": list(NAV_STATE_NAMES), "action": list(ACTION_NAMES),
                         "observation.goal_gps": ["goal_latitude", "goal_longitude"]}.get(key, names)
            if key == 'observation.gripper_effort_nm':
                names = ['left_gripper', 'right_gripper']
            if training_delivery:
                from .training_delivery import SIGNALS, COMMANDS, names as physical_names
                if key in SIGNALS or key in COMMANDS:
                    names = physical_names(key)
            features[key] = {"dtype": str(array.dtype), "shape": [array.shape[1]], "names": names}
            columns[key] = pa.array(array.tolist(), type=pa.list_(pa.from_numpy_dtype(array.dtype), array.shape[1]))
            stats[key] = _stats(array)
        scalars = {"timestamp": np.arange(count, dtype=np.float32) / fps,
                   "frame_index": np.arange(count, dtype=np.int64),
                   "episode_index": np.zeros(count, dtype=np.int64),
                   "index": np.arange(count, dtype=np.int64),
                   "task_index": np.zeros(count, dtype=np.int64),
                   "observation.capture_timestamp": capture}
        if all('captured_monotonic_s' in row for row in rows):
            stamps = np.asarray([row['captured_monotonic_s'] for row in rows], dtype=np.float64)
            if not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
                raise ValueError('Invalid shared observation timestamps')
            scalars['observation.capture_monotonic_s'] = stamps
        if continuous is not None:
            scalars.update(continuous['columns'])
        if omit_auxiliary_columns:
            scalars = {k: v for k, v in scalars.items()
                       if not k.startswith(('observation.capture_', 'observation.source_', 'action_source_'))}
        for key, values in scalars.items():
            features[key] = {"dtype": str(values.dtype), "shape": [1], "names": None}
            columns[key] = pa.array(values)
            stats[key] = _stats(values)
        data = staging / "data/chunk-000/file-000.parquet"
        data.parent.mkdir(parents=True)
        pq.write_table(pa.table(columns), data)
        # LeRobot tasks.parquet uses task text as the pandas index.
        tasks = pa.table({"task_index": pa.array([0], type=pa.int64()),
                          "__index_level_0__": [task]})
        pandas_metadata = {"index_columns": ["__index_level_0__"], "column_indexes": [],
            "columns": [
                {"name": "task_index", "field_name": "task_index", "pandas_type": "int64",
                 "numpy_type": "int64", "metadata": None},
                {"name": None, "field_name": "__index_level_0__", "pandas_type": "unicode",
                 "numpy_type": "object", "metadata": None}],
            "creator": {"library": "pyarrow", "version": pa.__version__}, "pandas_version": "2.2.3"}
        tasks = tasks.replace_schema_metadata({b"pandas": json.dumps(pandas_metadata).encode()})
        (staging / "meta").mkdir()
        pq.write_table(tasks, staging / "meta/tasks.parquet")
        episode = {"episode_index": 0, "tasks": [task], "length": count,
                   "dataset_from_index": 0, "dataset_to_index": count,
                   "data/chunk_index": 0, "data/file_index": 0,
                   "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0}
        if shared is not None:
            from .roboeval_shared import episode_annotation
            episode.update(episode_annotation(shared['outcome']))
        for camera in cameras:
            key = f"observation.images.{camera}"
            if continuous is not None:
                from .continuous_recording import encode_aligned_video
                video = staging / f"videos/{key}/chunk-000/file-000.mp4"
                video.parent.mkdir(parents=True)
                width, height = continuous['metadata']['source_resolution']
                stats[key] = encode_aligned_video(path, continuous['frames'][camera], video, fps)
                features[key] = {"dtype": "video", "shape": [height, width, 3],
                    "names": ["height", "width", "channels"],
                    "info": {"video.height": height, "video.width": width, "video.codec": "h264",
                             "video.pix_fmt": "yuv420p", "video.fps": fps, "video.channels": 3,
                             "video.is_depth_map": False, "has_audio": False}}
                episode.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
                                f"videos/{key}/from_timestamp": 0.,
                                f"videos/{key}/to_timestamp": count / fps})
                continue
            first_shape = None
            total = np.zeros(3)
            squares = np.zeros(3)
            minimum, maximum = np.ones(3), np.zeros(3)
            pixel_count = 0
            for row in rows:
                with Image.open(path / camera / f"{row['frame']:06d}.png") as image:
                    pixels = np.asarray(image.convert("RGB"))
                if first_shape is None:
                    first_shape = pixels.shape
                if pixels.shape != first_shape:
                    raise ValueError("Camera image shape changed within the trial")
                values = pixels.astype(np.float64).reshape(-1, 3) / 255
                total += values.sum(axis=0)
                squares += np.square(values).sum(axis=0)
                minimum = np.minimum(minimum, values.min(axis=0))
                maximum = np.maximum(maximum, values.max(axis=0))
                pixel_count += len(values)
            height, width, _ = first_shape
            # Padding handles odd camera sizes without changing source pixels.
            height += height % 2
            width += width % 2
            # Include padded black pixels in the normalization statistics.
            padding = count * (height * width - first_shape[0] * first_shape[1])
            if padding:
                minimum = np.minimum(minimum, 0)
                pixel_count += padding
            mean = total / pixel_count
            std = np.sqrt(np.maximum(squares / pixel_count - mean * mean, 0))
            stats[key] = {name: value.reshape(3, 1, 1).tolist() for name, value in
                          [("min", minimum), ("max", maximum), ("mean", mean), ("std", std)]}
            stats[key]["count"] = [count]
            video = staging / f"videos/{key}/chunk-000/file-000.mp4"
            video.parent.mkdir(parents=True)
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(fps), "-start_number", "0", "-i", str(path / camera / "%06d.png"),
                "-frames:v", str(count), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)],
                check=True)
            features[key] = {"dtype": "video", "shape": [height, width, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.height": height, "video.width": width, "video.codec": "h264",
                         "video.pix_fmt": "yuv420p", "video.fps": fps, "video.channels": 3,
                         "video.is_depth_map": False, "has_audio": False}}
            episode.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
                            f"videos/{key}/from_timestamp": 0.0,
                            f"videos/{key}/to_timestamp": count / fps})
        for key, values in stats.items():
            episode.update({f"stats/{key}/{name}": value for name, value in values.items()})
        ep_path = staging / "meta/episodes/chunk-000/file-000.parquet"
        ep_path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([episode]), ep_path)
        _json(staging / "meta/stats.json", stats)
        _json(staging / "meta/info.json", {
            "codebase_version": "v3.0", "robot_type": robot_type, "total_episodes": 1,
            "total_frames": count, "total_tasks": 1, "fps": fps,
            "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 500,
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features})
        stage_extension(extension, staging)
        if shared is not None:
            from .roboeval_shared import stage_shared
            stage_shared(shared, staging)
        files = {file.relative_to(staging).as_posix(): file_digest(file)
                 for file in staging.rglob("*") if file.is_file()}
        _json(staging / "meta/colosseum.json", {**context, "source_sha256": signature, "files": files})
        validate_uploaded_extension(staging)
        if extension is not None and shared is not None and len(cameras) == 3:
            if training_delivery:
                from .training_delivery import make_delivery as compact_dataset
            else:
                from .roboeval_compact import compact_dataset
            compact_staging = staging.with_name(staging.name + '.compact')
            compact_dataset(staging, compact_staging)
            # Both directories are generated by this export invocation only.
            shutil.rmtree(staging)
            staging = compact_staging
            validate_uploaded_extension(staging)
        if training_delivery and not (staging / 'meta/colosseum.json').exists():
            raise ValueError('Missing delivery metadata')
        if training_delivery:
            from .training_delivery import validate_delivery
            validate_delivery(staging)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if compact_staging is not None and compact_staging != staging and compact_staging.exists():
            shutil.rmtree(compact_staging)
    if export_destination is None:
        _clear_source_images(path, cameras, all_rows)
    return destination
