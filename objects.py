import numpy as np
import math
import json

# Default camera plane x vector
DEFAULT_X = np.array([1, 0, 0])
# Default camera direction
DEFAULT_Y = np.array([0, 1, 0])
# Default camera plane z vector
DEFAULT_Z = np.array([0, 0, 1])
# Default camera background color
BACKGROUND_DEFAULT_COLOR = np.array([20, 20, 20])
# Default object color
OBJECT_DEFAULT_COLOR = np.array([128, 128, 128])
# Default quaternion
DEFAULT_QUATERNION = np.array([0, 0, 1, 0])
# How much color varies by camera angle with object normals
COLOR_VARIANCE_FACTOR = 0.4
# How much should white contribute to color based on camera angle with object normals
SPECULAR_FACTOR = 0.4
# Degrees per radian
DPR = 180 / math.pi
# Small constant
EPSILON = 1e-5

min_raytrace_dist = 0

# Utility equations
diag = lambda q, a, b: 1 - 2 * (q[a] ** 2 + q[b] ** 2)
odiag_pos = lambda q, a, b, c, d: 2 * (q[a] * q[b] + q[c] * q[d])
odiag_neg = lambda q, a, b, c, d: 2 * (q[a] * q[b] - q[c] * q[d])
trig = lambda n, a: math.sin(n) if a else math.cos(n)
trig_3 = lambda v, a, b, c: trig(v[0], a) * trig(v[1], b) * trig(v[2], c)

# Converts a rotation quaternion into its corresponding conjugation matrix
def rot_quat_to_matrix(q):
    # assume unit quaternion
    matrix = [[diag(q, 2, 3), odiag_neg(q, 1, 2, 3, 0), odiag_pos(q, 1, 3, 2, 0)],
              [odiag_pos(q, 1, 2, 3, 0), diag(q, 1, 3), odiag_neg(q, 2, 3, 1, 0)],
              [odiag_neg(q, 1, 3, 2, 0), odiag_pos(q, 2, 3, 1, 0), diag(q, 1, 2)]]
    return np.array(matrix)

def rot_quat(axis, ang):
    unit_axis = axis / np.linalg.norm(axis)
    return np.concatenate((np.array([math.cos(ang/2)]), math.sin(ang/2)*unit_axis))

# Loads objects from JSON data
def load_objs(data):
    camera, results = None, []
    for loaded_data in data:
        if loaded_data["type"] == "Box":
            results.append(Box(position=np.array(loaded_data["position"]),
                              name=loaded_data["name"],
                              quaternion=np.array(loaded_data["quaternion"]),
                              color=np.array(loaded_data["color"]),
                              dims=np.array(loaded_data["dims"])))
        elif loaded_data["type"] == "Sphere":
            results.append(Sphere(position=np.array(loaded_data["position"]),
                               name=loaded_data["name"],
                               quaternion=np.array(loaded_data["quaternion"]),
                               color=np.array(loaded_data["color"]),
                               radius=loaded_data["radius"]))
        elif loaded_data["type"] == "Camera":
            if camera is None:
                camera = Camera(position=np.array(loaded_data["position"]),
                                  quaternion=np.array(loaded_data["quaternion"]),
                                  color=np.array(loaded_data["color"]),
                                  dims=np.array(loaded_data["dims"]),
                                  viewport_dims = np.array(loaded_data["vdims"]))
            else:
                raise TypeError("Cannot have more than one camera present in the scene.")
        else:
            raise TypeError("Invalid type of object found in JSON files!")
    if camera is None:
        raise TypeError("Camera is missing from the scene!")
    return camera, results

# An object with 3D space coordinates
class BaseObject:
    def __init__(self,
                 position=np.zeros(3),
                 name="object",
                 quaternion=DEFAULT_QUATERNION,
                 color=OBJECT_DEFAULT_COLOR):
        self.name = name
        self.position = position
        self.quaternion = quaternion
        self.color = color
        v = COLOR_VARIANCE_FACTOR / 2
        self.highColor = np.clip((1 + v) * self.color + SPECULAR_FACTOR * 255 * np.ones(3), 0, 255)
        self.midColor = np.clip(self.color + 0.5 * SPECULAR_FACTOR * 255 * np.ones(3), 0, 255)
        self.lowColor = np.clip((1 - v) * self.color, 0, 255)

    # Translates object by an offset of delta
    def translate(self, delta=np.zeros(3)):
        self.position += delta

    # Rotates object by offset quaternion around an position as a pivot
    def rotate(self, quaternion, pivot=None):
        pivot = pivot if pivot is not None else self.position
        delta = self.position - pivot
        # multiply rotation quaternions
        s, v = self.quaternion[0], self.quaternion[1:4]
        t, w = quaternion[0], quaternion[1:4]
        ang = s * t - np.dot(v.T, w)
        ax = s * w + t * v + np.cross(v, w)
        self.quaternion = np.concatenate((np.array([ang]), ax))
        # compute displacement from pivot
        if np.linalg.norm(delta) != 0:
            theta = 2 * math.acos(quaternion[0])
            axis = 2 * quaternion[1:4] / math.sin(theta)
            self.position = pivot + np.dot(rot_quat_to_matrix(rot_quat(axis, -theta)), delta)

    # Obtains the XYZ Euler angles from the object's quaternion
    def get_euler(self):
        q = self.quaternion
        rx = DPR * math.atan2(odiag_pos(q, 0, 1, 2, 3), diag(q, 1, 2))
        ry = DPR * math.asin(odiag_neg(q, 0, 2, 3, 1))
        rz = DPR * math.atan2(odiag_pos(q, 0, 3, 1, 2), diag(q, 2, 3))
        return np.array([(rx + 180) % 360, ry, (rz + 180) % 360])

    # Sets the object's quaternion according to input Euler angles
    def set_euler(self, euler=np.zeros(3)):
        self.quaternion = self.quaternion + EPSILON * np.zeros(4)
        e = 0.5 * (euler + np.array([180, 0, 180])) / DPR
        self.quaternion[0] = trig_3(e, 0, 0, 0) + trig_3(e, 1, 1, 1)
        self.quaternion[1] = trig_3(e, 1, 0, 0) - trig_3(e, 0, 1, 1)
        self.quaternion[2] = trig_3(e, 0, 1, 0) + trig_3(e, 1, 0, 1)
        self.quaternion[3] = trig_3(e, 0, 0, 1) - trig_3(e, 1, 1, 0)

    # Orthographic distance based on position point of object
    # The second result is additional context that may be used later
    def ortho_dist(self, origin, ray):
        return float("inf"), {}

    # Obtains matrix for performing a change of basis to object space
    def basis(self):
        return rot_quat_to_matrix(self.quaternion)

    # Returns a list of line data for rasterization
    def get_frame(self, camera):
        return []

    # BaseObjects have no tangible form and so do not show up on renders
    #     -1: object does not appear in path of ray
    #     0: object surface is parallel to ray
    #     1: object surface is perpendicular to ray
    def render(self, context):
        return -1

    # Obtains a color based on result of calling render
    def get_color_at(self, render):
        if (render < 0): # invalid call
            return np.array([0, 0, 0])
        if (render < 0.5):
            return 2 * render * self.midColor + (1 - 2 * render) * self.lowColor
        else:
            render -= 0.5
            return 2 * render * self.highColor + (1 - 2 * render) * self.midColor

    def update_colors(self):
        v = COLOR_VARIANCE_FACTOR / 2
        self.highColor = np.clip((1 + v) * self.color + SPECULAR_FACTOR * 255 * np.ones(3), 0, 255)
        self.midColor = np.clip(self.color + 0.5 * SPECULAR_FACTOR * 255 * np.ones(3), 0, 255)
        self.lowColor = np.clip((1 - v) * self.color, 0, 255)

    # Simple color fetch that bypasses rendering step
    def simple_color(self, context):
        return self.color

    # Returns the object as a JSON string for storage purposes
    def dict(self):
        return {
            "type": "BaseObject",
            "name": self.name,
            "position": self.position.tolist(),
            "quaternion": self.quaternion.tolist(),
            "color": self.color.tolist()
        }

# indices used to pair adjacent box corners into edges
BOX_CORNER_PAIR_IDXS = [(0, 4), (1, 5), (2, 6), (3, 7),
                        (0, 2), (1, 3), (4, 6), (5, 7),
                        (0, 1), (2, 3), (4, 5), (6, 7)]

# A box primitive 3D object
class Box(BaseObject):
    def __init__(self,
                 position=np.zeros(3),
                 name="box",
                 quaternion=DEFAULT_QUATERNION,
                 color=OBJECT_DEFAULT_COLOR,
                 dims=np.ones(3)):
        super().__init__(position, name, quaternion, color)
        self.dims = dims

    # Returns a list of line data for rasterization
    def get_frame(self, camera):
        # Collect corner points
        corners, frame = [], []
        I, J, K = 0.5 * self.dims
        for i in [-I, I]:
            for j in [-J, J]:
                for k in [-K, K]:
                    corners.append(np.array([i, j, k]))
        # Convert vectors to camera space
        to_world, to_camera = np.linalg.inv(self.basis()), camera.basis()
        corners = [to_camera @ (to_world @ corner + self.position - camera.position)
                   for corner in corners]
        # Assume consistent aspect ratio
        vp_ratio = camera.vdims[0] / camera.dims[0]
        corners = [np.array([vp_ratio*(c[0]+camera.dims[0]/2), c[1], vp_ratio*(c[2]+camera.dims[1]/2)])
                   for c in corners]
        delta_y = [c[1] for c in corners]
        diff = max(delta_y) - min(delta_y)
        get_color = lambda a, b: self.get_color_at(1 - (0.5 * (a + b) - min(delta_y)) / (max(1, diff)))
        return [(corners[a][[0, 2]], corners[b][[0, 2]], get_color(corners[a][1], corners[b][1]), "Line", (corners[a][1] + corners[b][1]) / 2)
                for a, b in BOX_CORNER_PAIR_IDXS]

    # Orthographic distance for a box
    def ortho_dist(self, origin, ray):
        # get box space coordinates of origin
        new_origin = np.dot(self.basis(), origin - self.position)
        ray = np.dot(self.basis(), ray)
        # obtain extent of box in box space
        pmin, pmax = -0.5 * self.dims, 0.5 * self.dims
        # bounding box check
        axes, tmins, tmaxs = [], [], []
        for i in range(3):
            if (ray[i] != 0):
                t0 = (pmin[i] - new_origin[i]) / ray[i]
                t1 = (pmax[i] - new_origin[i]) / ray[i]
                axes.append(i)
                tmins.append(min([t0, t1]))
                tmaxs.append(max([t0, t1]))
            elif new_origin[i] < pmin[i] or new_origin[i] > pmax[i]:
                return float("inf"), {}
        # ray misses box
        if max(tmins) > min(tmaxs):
            return float("inf"), {}
        # obtain time of first collision
        t, argmax_t = float("-inf"), -1
        for i, tmin in enumerate(tmins):
            if tmin > t:
                t = tmin
                argmax_t = axes[i]
        if t < 0:
            return float("inf"), {}
        return t * np.linalg.norm(ray), {
            "ray": ray / np.linalg.norm(ray),
            "argmax": argmax_t
        }

    # Simple color fetch that bypasses rendering step
    def simple_color(self, context):
        idxs = np.argsort(np.abs(context["ray"]))
        idxOf = np.where(idxs == context["argmax"])
        if min(idxOf[0]) == 2:
            return self.highColor
        elif min(idxOf[0]) == 1:
            return self.midColor
        return self.lowColor

    # Compute dot product between normal and ray
    #     based on what face the ray hit the box at
    # Because we are in object space, this is very simple
    def render(self, context):
        return abs(context["ray"][context["argmax"]])

    # Returns the box as a JSON string for storage purposes
    def dict(self):
        return {
            "type": "Box",
            "name": self.name,
            "position": self.position.tolist(),
            "quaternion": self.quaternion.tolist(),
            "color": self.color.tolist(),
            "dims": self.dims.tolist()
        }

# A sphere primitive 3D object
class Sphere(BaseObject):
    def __init__(self,
                 position=np.zeros(3),
                 name="sphere",
                 quaternion=DEFAULT_QUATERNION,
                 color=OBJECT_DEFAULT_COLOR,
                 radius=1):
        super().__init__(position, name, quaternion, color)
        self.radius = radius

    # Returns center and circumference data for rasterization
    def get_frame(self, camera):
        # Convert center vector to camera space
        frame = []
        center = camera.basis() @ (self.position - camera.position)
        # Assume consistent aspect ratio
        vp_ratio = camera.vdims[0] / camera.dims[0]
        center = np.array([vp_ratio * (center[0] + camera.dims[0] / 2), center[1], vp_ratio * (center[2] + camera.dims[1] / 2)])
        return [(center[[0, 2]], vp_ratio * self.radius, self.color, "Circle", center[1])]

    # Orthographic distance for a sphere
    def ortho_dist(self, origin, ray):
        # solve quadratic problem
        center = self.position
        a = np.dot(ray.T, ray)
        b = 2 * (np.dot((origin - center).T, ray))
        c = np.dot((origin - center).T, (origin - center)) - self.radius ** 2
        discrim = b ** 2 - 4 * a * c
        if discrim < 0: # ray misses sphere
            return float("inf"), {
                "ray": ray,
                "origin": origin,
                "dist": float("inf")
            }
        t0, t1 = (-b - discrim ** 0.5) / (2*a), (-b + discrim ** 0.5) / (2*a)
        # discard negative t values
        if t0 >= 0:
            return t0 * np.linalg.norm(ray), {
                "ray": ray,
                "origin": origin,
                "dist": t0
            }
        elif t1 >= 0:
            return t1 * np.linalg.norm(ray), {
                "ray": ray,
                "origin": origin,
                "dist": t1
            }
        else:
            return float("inf"), {
                "ray": ray,
                "origin": origin,
                "dist": float("inf")
            }

    # Compute dot product between incident ray and sphere surface normal
    def render(self, context):
        contact = context["origin"] + context["dist"] * context["ray"]
        normal = contact - self.position
        unit_normal = normal / np.linalg.norm(normal)
        unit_ray = context["ray"] / np.linalg.norm(context["ray"])
        return abs(np.dot(unit_normal.T, unit_ray))

    # Returns the box as a JSON string for storage purposes
    def dict(self):
        return {
            "type": "Sphere",
            "name": self.name,
            "position": self.position.tolist(),
            "quaternion": self.quaternion.tolist(),
            "color": self.color.tolist(),
            "radius": self.radius
        }

# A camera, represented as an object
class Camera(BaseObject):
    def __init__(self,
                 position=np.zeros(3),
                 quaternion=DEFAULT_QUATERNION,
                 color=BACKGROUND_DEFAULT_COLOR,
                 dims=np.array([4, 4]),
                 viewport_dims=np.array([240, 240])):
        super().__init__(position, "camera", quaternion, color)
        self.dims = dims
        self.vdims = viewport_dims

    # Generates the ingredients for an orthographic frame raster from a scene of objects
    def frame_rasterize(self, objs):
        frame_items = []
        for obj in objs:
            frame_items.extend(obj.get_frame(self))
        sort_fn = lambda itm: -itm[4]
        frame_items.sort(key=sort_fn)
        return [itm for itm in frame_items if itm[4] >= 0]

    # Generates an orthographic raytrace from a scene of objects
    def raytrace(self, objs, simple=False):
        I, J = self.vdims
        sheet = np.zeros((I, J, 3))
        defX, ray, defZ = self.basis()
        offX = - self.dims[0] * 0.5 * defX
        if simple:
            for i in range(I):
                offZ = -self.dims[0] * 0.5 * defZ
                for j in range(J):
                    origin = self.position + offX + offZ
                    # Compute orthographic dists for each object in scene
                    ortho_dists, contexts = [], []
                    for obj in objs:
                        dist, context = obj.ortho_dist(origin, ray)
                        if (dist != float("inf")):
                            ortho_dists.append((obj, dist))
                            contexts.append(context)
                    if len(ortho_dists) > 0:
                        idx = np.argmin(np.array([r[1] for r in ortho_dists]))
                        obj = ortho_dists[idx][0]
                        sheet[i, j] = np.round(obj.simple_color(contexts[idx]))
                    else:
                        sheet[i, j] = self.color
                    offZ += self.dims[0] / (J - 1) * defZ
                offX += self.dims[0] / (I - 1) * defX
        else:
            for i in range(I):
                offZ = -self.dims[0] * 0.5 * defZ
                for j in range(J):
                    origin = self.position + offX + offZ
                    # Compute orthographic dists for each object in scene
                    ortho_dists, contexts = [], []
                    for obj in objs:
                        dist, context = obj.ortho_dist(origin, ray)
                        if (dist != float("inf")):
                            ortho_dists.append((obj, dist))
                            contexts.append(context)
                    if len(ortho_dists) > 0:
                        idx = np.argmin(np.array([r[1] for r in ortho_dists]))
                        obj = ortho_dists[idx][0]
                        sheet[i, j] = np.round(obj.get_color_at(obj.render(contexts[idx])))
                    else:
                        sheet[i, j] = self.color
                    offZ += self.dims[0] / (J - 1) * defZ
                offX += self.dims[0] / (I - 1) * defX
        return sheet

    # Returns the camera as a JSON string for storage purposes
    def dict(self):
        return {
            "type": "Camera",
            "position": self.position.tolist(),
            "quaternion": self.quaternion.tolist(),
            "color": self.color.tolist(),
            "dims": self.dims.tolist(),
            "vdims": self.vdims.tolist()
        }

def test_camera_simple():
    camera = Camera(viewport_dims=np.array([12, 12]))
    sphere = Sphere(np.array([0, 2, 0]), name="sphere1",
                    quaternion=np.array([math.cos(math.pi / 8),
                                         math.sin(math.pi / 8) * math.sin(math.pi / 4),
                                         math.sin(math.pi / 8) * math.sin(math.pi / 4)]))
    box1 = Box(np.array([0, 2, 0]), name="box1",
               quaternion=np.array([math.cos(math.pi/6),
                                    math.sin(math.pi/6)*math.sin(math.pi/4),
                                    math.sin(math.pi/6)*math.sin(math.pi/4),
                                    0]),
               dims=np.array([2, 1, 3]))
    box2 = Box(np.array([0, 2, 0]), name="box2", dims=np.array([1, 2, 3]))
    print(camera.rasterize([box1])[:, :, 0])
    print(camera.rasterize([box2])[:, :, 0])
    print(camera.rasterize([sphere])[:, :, 0])

def test_camera_runtime():
    camera = Camera()
    sphere = Sphere(np.array([0, 2, 0]), name="sphere1",
                    quaternion=np.array([math.cos(math.pi / 8),
                                         math.sin(math.pi / 8) * math.sin(math.pi / 4),
                                         math.sin(math.pi / 8) * math.sin(math.pi / 4)]))
    box1 = Box(np.array([0, 2, 0]), name="box1",
               quaternion=np.array([math.cos(math.pi/6),
                                    math.sin(math.pi/6)*math.sin(math.pi/4),
                                    math.sin(math.pi/6)*math.sin(math.pi/4),
                                    0]),
               dims=np.array([2, 1, 3]))
    box2 = Box(np.array([0, 2, 0]), name="box2", dims=np.array([1, 2, 3]))
    print(camera.rasterize([box1, box2, sphere])[:, :, 0])

# test_camera_runtime()