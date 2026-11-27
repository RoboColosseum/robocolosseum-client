"""One calibration record per camera; factory stream extrinsics name both frames."""
import math


def factory_extrinsics(device, color_profile, rs):
    selected = {}
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if not profile.is_video_stream_profile():
                continue
            video = profile.as_video_stream_profile()
            if (video.width(), video.height(), profile.fps()) != (640, 480, 30):
                continue
            stream = profile.stream_type()
            if ((stream == rs.stream.depth and profile.format() == rs.format.z16)
                    or (stream == rs.stream.infrared and profile.format() == rs.format.y8)):
                selected[(str(stream), profile.stream_index())] = profile
    result = []
    for key, profile in sorted(selected.items()):
        ext = profile.get_extrinsics_to(color_profile)
        values = list(ext.rotation) + list(ext.translation)
        if len(values) != 12 or not all(math.isfinite(v) for v in values):
            raise ValueError('Invalid factory camera transform')
        result.append(dict(from_stream=key[0], from_stream_index=key[1],
            to_stream=str(color_profile.stream_type()), to_stream_index=color_profile.stream_index(),
            rotation_column_major=list(ext.rotation), translation_m=list(ext.translation),
            convention='p_to = R @ p_from + translation; SDK column-major 3x3 rotation',
            source='RealSense factory stream calibration', reference='within this device; not robot/world'))
    if not result:
        raise ValueError('No expected factory depth/IR-to-color calibration profiles')
    return result
