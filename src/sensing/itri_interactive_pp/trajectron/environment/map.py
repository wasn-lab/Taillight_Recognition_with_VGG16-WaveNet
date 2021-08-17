import torch
import numpy as np
from model.dataset.homography_warper import get_rotation_matrix2d, warp_affine_crop


class Map(object):
    def __init__(self, data, homography, description=None):
        self.data = data
        self.homography = homography
        self.description = description

    def as_image(self):
        raise NotImplementedError

    def get_cropped_maps(self, world_pts, patch_size, rotation=None, device='cpu'):
        raise NotImplementedError

    def to_map_points(self, scene_pts):
        raise NotImplementedError


class GeometricMap(Map):
    """
    A Geometric Map is a int tensor of shape [layers, x, y]. The homography must transform a point in scene
    coordinates to the respective point in map coordinates.

    :param data: Numpy array of shape [layers, x, y]
    :param homography: Numpy array of shape [3, 3]
    """
    def __init__(self, data, homography, description=None):
        #assert isinstance(data.dtype, np.floating), "Geometric Maps must be float values."
        super(GeometricMap, self).__init__(data, homography, description=description)

        self._last_padding = None
        self._last_padded_map = None
        self._torch_map = None

    def torch_map(self, device):
        if self._torch_map is not None:
            return self._torch_map
        self._torch_map = torch.tensor(self.data, dtype=torch.uint8, device=device)
        return self._torch_map

    def as_image(self):
        # We have to transpose x and y to rows and columns. Assumes origin is lower left for image
        # Also we move the channels to the last dimension
        return (np.transpose(self.data, (2, 1, 0))).astype(np.uint)

    def get_padded_map(self, padding_x, padding_y, device):
        if self._last_padding == (padding_x, padding_y):
            return self._last_padded_map
        else:
            self._last_padding = (padding_x, padding_y)
            # print("get_padded_map data.shape[0]", self.data.shape[0])
            # print("get_padded_map data.shape[1]", self.data.shape[1])
            # print("get_padded_map data.shape[1]", self.data.shape[2])
            self._last_padded_map = torch.full((self.data.shape[0],
                                                self.data.shape[1] + 2 * padding_x,
                                                self.data.shape[2] + 2 * padding_y),
                                               False, dtype=torch.uint8)
            self._last_padded_map[..., padding_x:-padding_x, padding_y:-padding_y] = self.torch_map(device)
            return self._last_padded_map

    @staticmethod
    def batch_rotate(map_batched, centers, angles, out_height, out_width):
        """
        As the input is a map and the warp_affine works on an image coordinate system we would have to
        flip the y axis updown, negate the angles, and flip it back after transformation.
        This, however, is the same as not flipping at and not negating the radian.

        :param map_batched:
        :param centers:
        :param angles:
        :param out_height:
        :param out_width:
        :return:
        """
        M = get_rotation_matrix2d(centers, angles, torch.ones_like(angles))
        # print("map batch_rotate", map_batched.shape)
        rotated_map_batched = warp_affine_crop(map_batched, centers, M,
                                               dsize=(out_height, out_width), flags='bilinear', padding_mode='zeros')

        return rotated_map_batched

    @classmethod
    def get_cropped_maps_from_scene_map_batch(cls, maps, scene_pts, patch_size, rotation=None, device='cpu'):
        """
        Returns rotated patches of each map around the transformed scene points.
        ___________________
        |       |          |
        |       |ps[3]     |
        |       |          |
        |       |          |
        |      o|__________|
        |       |    ps[2] |
        |       |          |
        |_______|__________|
        ps = patch_size

        :param maps: List of GeometricMap objects [bs]
        :param scene_pts: Scene points: [bs, 2]
        :param patch_size: Extracted Patch size after rotation: [-x, -y, +x, +y]
        :param rotation: Rotations in degrees: [bs]
        :param device: Device on which the rotated tensors should be returned.
        :return: Rotated and cropped tensor patches.
        """
        # For ITRIADV map: map_origin = torch.tensor([[2008.5, 41172.9]])
        # For ZhuBei map: map_origin = torch.tensor([[633.0, 44853.9]])
        map_origin = torch.tensor([[2008.5, 41172.9]])
        
        scene_pts = scene_pts - map_origin
        batch_size = scene_pts.shape[0]
        
        # print("scene_pts", scene_pts)
        # print("batch_size", batch_size)
        lat_size = 2 * np.max((patch_size[0], patch_size[2]))
        long_size = 2 * np.max((patch_size[1], patch_size[3]))
        assert lat_size % 2 == 0, "Patch width must be divisible by 2"
        assert long_size % 2 == 0, "Patch length must be divisible by 2"
        lat_size_half = lat_size // 2
        long_size_half = long_size // 2

        context_padding_x = int(np.ceil(np.sqrt(2) * lat_size))
        context_padding_y = int(np.ceil(np.sqrt(2) * long_size))
        # print("context_padding_x", context_padding_x)
        # print("context_padding_y", context_padding_y)
        # print("centers part I", scene_pts)
        # print("centers part II", torch.tensor([context_padding_x, context_padding_y], device=device, dtype=torch.long))
        
        # to_map_points
        # centers = torch.tensor([s_map.to_map_points(scene_pts[np.newaxis, i]) for i, s_map in enumerate(maps)],
        #                        dtype=torch.long, device=device).squeeze(dim=1) \
        #           + torch.tensor([context_padding_x, context_padding_y], device=device, dtype=torch.long)
        centers = scene_pts \
                  + torch.tensor([context_padding_x, context_padding_y], device=device, dtype=torch.long)
        # print("centers shape", centers.shape)
        # print("centers", centers)

        # get_padded_map
        padded_map = [s_map.get_padded_map(context_padding_x, context_padding_y, device=device) for s_map in maps]
        # print("padded_map len", len(padded_map))
        # print("padded_map[0].shape ", padded_map[0].shape)
        # print("padded_map[0]", padded_map[0])
        
        for i in range(centers.shape[0]):
            # print(i)
            # print("padded_map_batched X-padding: ", padded_map[i].shape[1])
            # print("padded_map_batched Y-padding: ", padded_map[i].shape[2])
            # print("centers.x - padding_x: ", int(centers[i, 0] - context_padding_x))
            # print("centers.x + padding_x: ", int(centers[i, 0] + context_padding_x))
            # print("centers.y - padding_y: ", int(centers[i, 1] - context_padding_y))
            # print("centers.y + padding_y: ", int(centers[i, 1] + context_padding_y))
            
            # pad the padded map if x or y exceed the map to avoid runtime dimension mismatch error
            if int(centers[i, 0] + context_padding_x) > padded_map[i].shape[1]:
                p_size = int(centers[i, 0] + context_padding_x) - padded_map[i].shape[1]
                padder = torch.zeros(3, p_size, padded_map[i].shape[2], dtype=torch.uint8)
                padded_map[i] = torch.cat((padded_map[i], padder), dim = 1)
            if int(centers[i, 1] + context_padding_y) > padded_map[i].shape[2]:
                p_size = int(centers[i, 1] + context_padding_y) - padded_map[i].shape[2]
                padder = torch.zeros(3, padded_map[i].shape[1], p_size, dtype=torch.uint8)
                print("Ytype of padded_map[i]", type(padded_map[i]))
                print("Ytype of padder[i]", type(padder[i]))
                padded_map[i] = torch.cat((padded_map[i], padder), dim = 2)
        
        # print("padded_map len", len(padded_map))
        # print("padded_map[0].shape ", padded_map[0].shape)
        
        padded_map_batched = torch.stack([padded_map[i][Ellipsis,
                                          int(centers[i, 0] - context_padding_x): int(centers[i, 0] + context_padding_x),
                                          int(centers[i, 1] - context_padding_y): int(centers[i, 1] + context_padding_y)]
                                          for i in range(centers.shape[0])], dim=0)
        
        # 0 appears here
        # print("padded_map_batched shape", padded_map_batched.shape)
        center_patches = torch.tensor([[context_padding_y, context_padding_x]],
                                      dtype=torch.int,
                                      device=device).repeat(batch_size, 1)

        if rotation is not None:
            angles = torch.Tensor(rotation)
        else:
            angles = torch.zeros(batch_size)
        
        rotated_map_batched = cls.batch_rotate(padded_map_batched/255.,
                                                center_patches.float(),
                                                angles,
                                                long_size,
                                                lat_size)

        del padded_map_batched

        return rotated_map_batched[...,
               long_size_half - patch_size[1]:(long_size_half + patch_size[3]),
               lat_size_half - patch_size[0]:(lat_size_half + patch_size[2])]

    def get_cropped_maps(self, scene_pts, patch_size, rotation=None, device='cpu'):
        """
        Returns rotated patches of the map around the transformed scene points.
        ___________________
        |       |          |
        |       |ps[3]     |
        |       |          |
        |       |          |
        |      o|__________|
        |       |    ps[2] |
        |       |          |
        |_______|__________|
        ps = patch_size

        :param scene_pts: Scene points: [bs, 2]
        :param patch_size: Extracted Patch size after rotation: [-lat, -long, +lat, +long]
        :param rotation: Rotations in degrees: [bs]
        :param device: Device on which the rotated tensors should be returned.
        :return: Rotated and cropped tensor patches.
        """
        return self.get_cropped_maps_from_scene_map_batch([self]*scene_pts.shape[0], scene_pts,
                                                          patch_size, rotation=rotation, device=device)

    def to_map_points(self, scene_pts):
        org_shape = None
        if len(scene_pts.shape) > 2:
            org_shape = scene_pts.shape
            scene_pts = scene_pts.reshape((-1, 2))
        N, dims = scene_pts.shape
        points_with_one = np.ones((dims + 1, N))
        points_with_one[:dims] = scene_pts.T
        # python 3
        # map_points = (self.homography @ points_with_one).T[..., :dims]

        # python2
        map_points = np.matmul(self.homography, points_with_one).T[..., :dims]
        if org_shape is not None:
            map_points = map_points.reshape(org_shape)
        return map_points

class ImageMap(Map):  # TODO Implement for image maps -> watch flipped coordinate system
    def __init__(self):
        raise NotImplementedError