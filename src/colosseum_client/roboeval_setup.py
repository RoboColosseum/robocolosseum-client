"""Validated setup and offline kinematics; importing this module opens no devices."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ARM_IDS = ('left', 'right')
SELECTED_FIELDS = [*(f'A{i}' for i in range(1, 10)), 'B1', 'B2', 'B3', 'B5', 'B6',
                   'C1', 'C2', 'C3', 'C4', 'C5', 'C8']


def array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    return result


def transform(value, name):
    result = array(value, (4, 4), name)
    rotation = result[:3, :3]
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise ValueError(f'{name} must be a rigid transform')
    return result


class Setup:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.document = json.loads(self.path.read_text(encoding='utf-8'))
        doc = self.document
        if doc.get('calibrated') is not True or not doc.get('calibration_version'):
            raise ValueError('RoboEval setup requires verified calibration; fill the setup template first')
        self.arms = doc.get('arms')
        if (not isinstance(self.arms, dict) or not self.arms
                or any(name not in ARM_IDS for name in self.arms)):
            raise ValueError('RoboEval arms must be keyed by left/right')
        self.arms = {side: self.arms[side] for side in ARM_IDS if side in self.arms}
        for name, arm in self.arms.items():
            transform(arm.get('base_to_world'), name + '.base_to_world')
            if arm.get('camera_to_eef') is not None:
                transform(arm['camera_to_eef'], name + '.camera_to_eef')
            if not isinstance(arm.get('eef_site'), str) or not arm['eef_site']:
                raise ValueError(f'{name}.eef_site is required')
            mapping = arm.get('gripper', {})
            readings = np.asarray(mapping.get('readings'), dtype=float)
            widths = np.asarray(mapping.get('widths_m'), dtype=float)
            if (readings.ndim != 1 or len(readings) < 2 or readings.shape != widths.shape
                    or not np.isfinite(readings).all() or not np.isfinite(widths).all()
                    or np.any(np.diff(readings) <= 0) or np.any(np.diff(widths) <= 0)
                    or np.any(widths < 0) or readings[0] > 0 or readings[-1] < 1):
                raise ValueError(f'{name}.gripper requires increasing readings and measured widths_m covering [0, 1]')
        self.model_path = self.resolve(doc.get('model_path'))
        self.urdf_path = self.resolve(doc.get('urdf_path'))
        for path in (self.model_path, self.urdf_path):
            if not path.is_file():
                raise ValueError(f'RoboEval model is missing: {path}')

    def validate_cameras(self, cameras):
        for name, serial in cameras.items():
            side = name.removesuffix('_image')
            if side in self.arms:
                if str(self.arms[side].get('camera_serial')) != str(serial):
                    raise ValueError(f'{name} serial differs from verified setup')
            else:
                fixed = self.document.get('fixed_cameras', {}).get(name, {})
                if fixed.get('camera_to_world') is not None:
                    transform(fixed['camera_to_world'], name + '.camera_to_world')
                if str(fixed.get('serial')) != str(serial):
                    raise ValueError(f'{name} serial differs from verified setup')
        if any(f'{side}_image' not in cameras for side in self.arms):
            raise ValueError('Each recorded arm requires its wrist camera')

    def resolve(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError('RoboEval model_path and urdf_path are required')
        return (self.path.parent / value).resolve()

    def width(self, side, reading):
        mapping = self.arms[side]['gripper']
        x, y = np.asarray(mapping['readings']), np.asarray(mapping['widths_m'])
        i = int(np.clip(np.searchsorted(x, reading) - 1, 0, len(x) - 2))
        # Keep endpoint overshoot: do not turn clipping into a measured value.
        return float(y[i] + (reading - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]))


class Kinematics:
    def __init__(self, setup):
        import mujoco
        self.mj, self.setup = mujoco, setup
        self.model = mujoco.MjModel.from_xml_path(str(setup.model_path))
        self.data = {name: mujoco.MjData(self.model) for name in setup.arms}
        self.indices, self.sites = {}, {}
        for side, arm in setup.arms.items():
            names = arm.get('joint_names', [f'joint{i}' for i in range(1, 7)])
            if len(names) != 6:
                raise ValueError('YAM RoboEval requires six arm joint names')
            self.indices[side] = []
            for name in names:
                joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint < 0:
                    raise ValueError(f'Model joint not found: {name}')
                self.indices[side].append(int(self.model.jnt_qposadr[joint]))
            site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, arm['eef_site'])
            if site < 0:
                raise ValueError(f'Model eef_site not found: {arm["eef_site"]}')
            self.sites[side] = site

    def pose(self, side, joints):
        data = self.data[side]
        data.qpos[self.indices[side]] = array(joints, (6,), 'joint_position_rad')
        self.mj.mj_forward(self.model, data)
        site = self.sites[side]
        base = np.asarray(self.setup.arms[side]['base_to_world'])
        position = base[:3, :3] @ data.site_xpos[site] + base[:3, 3]
        rotation = base[:3, :3] @ data.site_xmat[site].reshape(3, 3)
        quaternion = np.empty(4)
        self.mj.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation).ravel())
        quaternion /= np.linalg.norm(quaternion)
        return position.tolist(), quaternion.tolist()


def copy_model(source, destination):
    """Copy XML/URDF and its referenced resources, rewriting to portable paths."""
    source, destination = Path(source).resolve(), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source)
    compiler = tree.find('compiler')
    directories = {} if compiler is None else dict(compiler.attrib)
    if compiler is not None:
        compiler.attrib.pop('meshdir', None)
        compiler.attrib.pop('texturedir', None)
    for element in tree.iter():
        for attr in ('file', 'filename'):
            reference = element.get(attr)
            if not reference:
                continue
            asset = Path(reference)
            if reference.startswith('package://'):
                candidates = list(source.parent.rglob(Path(reference).name))
                if len(candidates) != 1:
                    raise ValueError(f'Cannot resolve model resource: {reference}')
                asset = candidates[0]
            elif not asset.is_absolute():
                directory = directories.get('meshdir', '') if element.tag == 'mesh' else directories.get('texturedir', '') if element.tag == 'texture' else ''
                asset = source.parent / directory / asset
            asset = asset.resolve()
            if not asset.is_file():
                raise ValueError(f'Missing model resource: {asset}')
            digest = hashlib.sha256(asset.read_bytes()).hexdigest()
            relative = Path('assets') / (digest[:20] + asset.suffix.lower())
            target = destination.parent / relative
            target.parent.mkdir(exist_ok=True)
            if element.tag == 'include':
                copy_model(asset, target)
            elif not target.exists():
                target.write_bytes(asset.read_bytes())
            element.set(attr, relative.as_posix())
    tree.write(destination, encoding='utf-8', xml_declaration=True)


def setup_snapshot(setup, root, *, cameras, identity, limits):
    root = Path(root)
    copy_model(setup.model_path, root / 'models/model.xml')
    copy_model(setup.urdf_path, root / 'models/robot.urdf')
    doc = copy.deepcopy(setup.document)
    doc.update(model_path='models/model.xml', urdf_path='models/robot.urdf',
               selected_fields=SELECTED_FIELDS, cameras=cameras, identity=identity,
               runtime_limits=limits, quaternion_order='w,x,y,z',
               gripper_convention='meters; larger means more open')
    for arm in doc['arms'].values():
        arm['gripper'].update(min_width_m=arm['gripper']['widths_m'][0],
                              max_width_m=arm['gripper']['widths_m'][-1],
                              direction='larger means more open')
    doc['stream_semantics'] = dict(
        state_clock='host monotonic seconds; receipt of a complete sequential motor sweep',
        arm_order=list(setup.arms),
        joint_action_values='joint1..6 radians, normalized gripper (larger means more open)',
        raw_policy='complete postprocessed model output before action safety clipping; may exceed executable prefix',
        sent_command='target accepted by SDK command_joint_pos; not measured achievement',
        video_clock='video frame index / 30 within each segment; source/receipt timestamps in frames.parquet are authoritative',
        factory_camera_extrinsic='named source stream to named target stream within one device; not robot/world',
        fixed_camera_extrinsic='optional supplied installation geometry: camera coordinates to shared world',
        wrist_camera_extrinsic='optional supplied installation geometry: camera coordinates to configured end-effector site')
    doc['urdf_declared_joint_limits'] = {
        node.get('name'): dict(node.find('limit').attrib)
        for node in ET.parse(setup.urdf_path).getroot().findall('joint')
        if node.find('limit') is not None}
    doc['limit_semantics'] = ('URDF declarations are preserved verbatim, including upstream placeholders. '
        'Runtime joint bounds and motor protocol torque/velocity envelopes are saved separately; '
        'they do not certify firmware register settings or a physical safety limit.')
    (root / 'setup.json').write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_template(destination, cameras=None):
    """Prepare model resources without constructing a robot or opening CAN."""
    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or (destination.parent / 'models').exists():
        raise FileExistsError(destination)
    model = Path(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310))
    urdf = Path(ArmType.YAM.get_xml_path()).with_suffix('.urdf')
    try:
        copy_model(model, destination.parent / 'models/model.xml')
        copy_model(urdf, destination.parent / 'models/robot.urdf')
    finally:
        model.unlink(missing_ok=True)  # Only the SDK temporary file created by this call.
    doc = dict(calibrated=False, calibration_version='', robot_model='I2RT YAM / LINEAR_4310',
               model_path='models/model.xml', urdf_path='models/robot.urdf', arms={}, fixed_cameras={})
    cameras = cameras or {}
    for side in ARM_IDS:
        doc['arms'][side] = dict(base_to_world=None, camera_to_eef=None, eef_site='grasp_site',
            camera_serial=cameras.get(f'{side}_image'),
            joint_names=[f'joint{i}' for i in range(1, 7)],
            gripper=dict(readings=[0, 1], widths_m=[None, None]))
    for name, serial in cameras.items():
        if name not in ('left_image', 'right_image'):
            doc['fixed_cameras'][name] = dict(serial=serial, camera_to_world=None)
    destination.write_text(json.dumps(doc, indent=2) + '\n', encoding='utf-8')
    return destination


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare or validate RoboEval setup without opening devices')
    parser.add_argument('setup')
    parser.add_argument('--template', action='store_true')
    parser.add_argument('--config', help='Read camera identities from an existing Client YAML, without opening devices')
    args = parser.parse_args()
    if args.template:
        cameras = None
        if args.config:
            import yaml
            cameras = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))['cameras']
        print(write_template(args.setup, cameras))
    else:
        Kinematics(Setup(args.setup))
        print('Setup and offline kinematics are valid; no devices were opened.')


if __name__ == '__main__':
    main()
