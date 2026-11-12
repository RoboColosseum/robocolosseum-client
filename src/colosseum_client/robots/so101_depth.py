"""Metric depth aligned to a LeRobot RealSense color stream, for policies that ask for it.

Only the local Policy Server receives it (as observation state tensors); the
Client's recordings and dataset exports never include depth.
"""
import numpy as np
from PIL import Image


class AlignedDepth:
    """Align one RealSense camera's depth to its color image and report metres.

    ``install`` wraps the camera's frame read with ``rs.align(rs.stream.color)``
    so each depth pixel indexes the same color pixel, then records the depth
    scale and the color intrinsics rescaled to the Client's image size.
    """

    def __init__(self, role, image_size):
        self.role = role
        self.image_size = tuple(image_size)  # (width, height) sent to the policy
        self.scale = None
        self.intrinsics = None

    def install(self, camera, *, rs=None):
        if not getattr(camera, 'use_depth', False):
            raise RuntimeError(f'SO101 camera {self.role} has no depth stream')
        if getattr(camera, 'rotation', None) is not None:
            raise RuntimeError('depth alignment requires an unrotated RealSense stream')
        if rs is None:
            import pyrealsense2 as rs
        if not getattr(camera, '_colosseum_depth_aligned', False):
            align = rs.align(rs.stream.color)
            read = camera._read_from_hardware
            camera._read_from_hardware = lambda: align.process(read())
            camera._colosseum_depth_aligned = True
        profile = camera.rs_profile
        self.scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        k = stream.get_intrinsics()
        sx = self.image_size[0] / stream.width()
        sy = self.image_size[1] / stream.height()
        self.intrinsics = np.array(
            [[k.fx * sx, 0.0, k.ppx * sx], [0.0, k.fy * sy, k.ppy * sy], [0.0, 0.0, 1.0]],
            dtype=np.float32)

    def read(self, camera):
        """Return (depth_m HxW float32, 3x3 intrinsics) from the latest buffered frame.

        Reads the buffer under the camera's lock without waiting on its
        new-frame event, so color reads for the same observation are unaffected.
        """
        if self.scale is None:
            raise RuntimeError('depth is unavailable before install()')
        with camera.frame_lock:
            raw = camera.latest_depth_frame
        if raw is None:
            raise RuntimeError(f'SO101 camera {self.role} has no depth frame yet')
        depth = np.asarray(raw).reshape(np.asarray(raw).shape[:2]).astype(np.float32) * np.float32(self.scale)
        if (depth.shape[1], depth.shape[0]) != self.image_size:
            depth = np.asarray(Image.fromarray(depth).resize(
                self.image_size, resample=Image.Resampling.NEAREST), dtype=np.float32)
        return np.ascontiguousarray(depth), self.intrinsics.copy()
