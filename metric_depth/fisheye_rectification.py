"""Double-sphere left-view rectification for the multi-camera pair YAML format.

Uses the static (zero-velocity) plane and axis conventions in
/home/share/ray/bags/rectify/rectify_ds_cam_pairs.py and rectify_utils.py.
Inverse remapping samples that plane directly, avoiding per-frame splatting.
"""
import cv2
import numpy as np
import yaml


class DoubleSphereRectifier:
    def __init__(self, calibration, pair='2_1', size=320, fov=90.0):
        with open(calibration) as stream:
            config = yaml.safe_load(stream)
        cameras = config[f'cam_pair_{pair}']
        camera = cameras['cam0']
        if camera['camera_model'] != 'ds':
            raise ValueError('Rectification requires double-sphere (ds) calibration')
        self.source_size = tuple(camera['resolution'])
        self.size = size
        if size < 2 or not 0 < fov < 180:
            raise ValueError('Rectified size must be >=2 and FOV between 0 and 180 degrees')
        xi, alpha, fx, fy, cx, cy = camera['intrinsics']
        # Preserve the existing pair rectifier's extrinsic convention.
        rotation = np.asarray(cameras['cam1']['R'], dtype=np.float64).T
        translation = rotation.T @ np.asarray(cameras['cam1']['t'], dtype=np.float64)
        normal = np.array([0., 0., 1.]) + rotation[:, 2]
        normal -= np.dot(normal, translation) / np.dot(translation, translation) * translation
        normal /= np.linalg.norm(normal)
        u_axis = -translation / np.linalg.norm(translation)
        v_axis = np.cross(normal, u_axis)
        v_axis /= np.linalg.norm(v_axis)
        half_angle = np.deg2rad(fov / 2)
        grid = np.linspace(-np.tan(half_angle), np.tan(half_angle), size)
        u, v = np.meshgrid(grid, grid)
        # Existing plane projection intersects n.X = -1, so invert its signs.
        rays = normal - u[..., None] * u_axis - v[..., None] * v_axis
        x, y, z = np.moveaxis(rays, -1, 0)
        d1 = np.linalg.norm(rays, axis=-1)
        z1 = xi * d1 + z
        d2 = np.sqrt(x*x + y*y + z1*z1)
        denominator = alpha * d2 + (1-alpha) * z1
        self.map_x = (fx * x / denominator + cx).astype(np.float32)
        self.map_y = (fy * y / denominator + cy).astype(np.float32)
        width, height = self.source_size
        # The reference rectifier limits rays to a circular FOV before splatting.
        valid = ((rays @ normal) / d1 > np.cos(half_angle)) & (denominator > 1e-12)
        valid &= np.isfinite(self.map_x) & np.isfinite(self.map_y)
        valid &= (self.map_x >= 0) & (self.map_x < width-1) & (self.map_y >= 0) & (self.map_y < height-1)
        self.valid = valid
        self.map_x[~valid] = -1
        self.map_y[~valid] = -1
        self.metadata = {'calibration': str(calibration), 'pair': pair,
                         'source_size': self.source_size, 'output_size': [size, size],
                         'fov_degrees': fov, 'normal': normal.tolist(),
                         'u_axis': u_axis.tolist(), 'v_axis': v_axis.tolist(),
                         'method': 'double_sphere_inverse_remap_static_plane',
                         'valid_fraction': float(valid.mean())}

    def __call__(self, bgr):
        if bgr.shape[1::-1] != self.source_size:
            raise ValueError(f'Image size {bgr.shape[1::-1]} differs from calibration {self.source_size}; provide matching calibration')
        return cv2.remap(bgr, self.map_x, self.map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
