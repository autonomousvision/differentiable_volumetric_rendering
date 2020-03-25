import torch
import torch.nn as nn
from torch import distributions as dist
from im2mesh.dvr.models import (
    decoder, depth_function
)
from im2mesh.common import (
    get_mask, image_points_to_world, origin_to_world, normalize_tensor)


# Decoder dictionary
decoder_dict = {
    'simple': decoder.Decoder,
}


class DVR(nn.Module):
    ''' DVR model class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
        depth_function_kwargs (dict): keyworded arguments for the
            depth_function
    '''

    def __init__(self, decoder, encoder=None,
                 device=None, depth_function_kwargs={}):
        super().__init__()
        self.decoder = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device
        self.call_depth_function = depth_function.DepthModule(
            **depth_function_kwargs)

    def forward(self, pixels, p_occupancy, p_freespace, inputs, camera_mat,
                world_mat, scale_mat, it=None,  sparse_depth=None,
                calc_normals=False, **kwargs):
        ''' Performs a forward pass through the network.

        This function evaluates the depth and RGB color values for respective
        points as well as the occupancy values for the points of the helper
        losses. By wrapping everything in the forward pass, multi-GPU training
        is enabled.

        Args:
            pixels (tensor): sampled pixels
            p_occupancy (tensor): points for occupancy loss
            p_freespace (tensor): points for freespace loss
            inputs (tensor): input
            camera_mat (tensor): camera matrices
            world_mat (tensor): world matrices
            scale_mat (tensor): scale matrices
            it (int): training iteration (used for ray sampling scheduler)
            sparse_depth (dict): if not None, dictionary with sparse depth data
            calc_normals (bool): whether to calculate normals for surface
                points and a randomly-sampled neighbor
        '''
        # encode inputs
        c = self.encode_inputs(inputs)

        # transform pixels p to world
        p_world, mask_pred, mask_zero_occupied = \
            self.pixels_to_world(pixels, camera_mat,
                                 world_mat, scale_mat, c, it)
        rgb_pred = self.decode_color(p_world, c=c)

        # eval occ at sampled p
        logits_occupancy = self.decode(p_occupancy, c=c).logits

        # eval freespace at p and
        # fill in predicted world points
        p_freespace[mask_pred] = p_world[mask_pred].detach()
        logits_freespace = self.decode(p_freespace, c=c,).logits

        if calc_normals:
            normals = self.get_normals(p_world.detach(), mask_pred, c=c)
        else:
            normals = None

        # Project pixels for sparse depth loss to world if dict is not None
        if sparse_depth is not None:
            p = sparse_depth['p']
            camera_mat = sparse_depth['camera_mat']
            world_mat = sparse_depth['world_mat']
            scale_mat = sparse_depth['scale_mat']
            p_world_sparse, mask_pred_sparse, _ = self.pixels_to_world(
                p, camera_mat, world_mat, scale_mat, c, it)
        else:
            p_world_sparse, mask_pred_sparse = None, None

        return (p_world, rgb_pred, logits_occupancy, logits_freespace,
                mask_pred, p_world_sparse, mask_pred_sparse, normals)

    def get_normals(self, points, mask, c=None, h_sample=1e-3,
                    h_finite_difference=1e-3):
        ''' Returns the unit-length normals for points and one randomly
        sampled neighboring point for each point.

        Args:
            points (tensor): points tensor
            mask (tensor): mask for points
            c (tensor): latent conditioned code c
            h_sample (float): interval length for sampling the neighbors
            h_finite_difference (float): step size finite difference-based
                gradient calculations
        '''
        device = self._device

        if mask.sum() > 0:
            c = c.unsqueeze(1).repeat(1, points.shape[1], 1)[mask]
            points = points[mask]
            points_neighbor = points + (torch.rand_like(points) * h_sample -
                                        (h_sample / 2.))

            normals_p = normalize_tensor(
                self.get_central_difference(points, c=c,
                                            h=h_finite_difference))
            normals_neighbor = normalize_tensor(
                self.get_central_difference(points_neighbor, c=c,
                                            h=h_finite_difference))
        else:
            normals_p = torch.empty(0, 3).to(device)
            normals_neighbor = torch.empty(0, 3).to(device)

        return [normals_p, normals_neighbor]

    def get_central_difference(self, points, c=None, h=1e-3):
        ''' Calculates the central difference for points.

        It approximates the derivative at the given points as follows:
            f'(x) ≈ f(x + h/2) - f(x - h/2) for a small step size h

        Args:
            points (tensor): points
            c (tensor): latent conditioned code c
            h (float): step size for central difference method
        '''
        n_points, _ = points.shape
        device = self._device

        if c.shape[-1] != 0:
            c = c.unsqueeze(1).repeat(1, 6, 1).view(-1, c.shape[-1])

        # calculate steps x + h/2 and x - h/2 for all 3 dimensions
        step = torch.cat([
            torch.tensor([1., 0, 0]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([-1., 0, 0]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([0, 1., 0]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([0, -1., 0]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([0, 0, 1.]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([0, 0, -1.]).view(1, 1, 3).repeat(n_points, 1, 1)
        ], dim=1).to(device) * h / 2
        points_eval = (points.unsqueeze(1).repeat(1, 6, 1) + step).view(-1, 3)

        # Eval decoder at these points
        f = self.decoder(points_eval, c=c, only_occupancy=True,
                         batchwise=False).view(n_points, 6)

        # Get approximate derivate as f(x + h/2) - f(x - h/2)
        df_dx = torch.stack([
            (f[:, 0] - f[:, 1]),
            (f[:, 2] - f[:, 3]),
            (f[:, 4] - f[:, 5]),
        ], dim=-1)
        return df_dx

    def decode(self, p, c=None, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c, only_occupancy=True, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def march_along_ray(self, ray0, ray_direction, c=None, it=None,
                        sampling_accuracy=None):
        ''' Marches along the ray and returns the d_i values in the formula
            r(d_i) = ray0 + ray_direction * d_i
        which returns the surfaces points.

        Here, ray0 and ray_direction are directly used without any
        transformation; Hence the evaluation is done in object-centric
        coordinates.

        Args:
            ray0 (tensor): ray start points (camera centers)
            ray_direction (tensor): direction of rays; these should be the
                vectors pointing towards the pixels
            c (tensor): latent conditioned code c
            it (int): training iteration (used for ray sampling scheduler)
            sampling_accuracy (tuple): if not None, this overwrites the default
                sampling accuracy ([128, 129])
        '''
        device = self._device

        d_i = self.call_depth_function(ray0, ray_direction, self.decoder,
                                       c=c, it=it, n_steps=sampling_accuracy)

        # Get mask for where first evaluation point is occupied
        mask_zero_occupied = d_i == 0

        # Get mask for predicted depth
        mask_pred = get_mask(d_i).detach()

        # For sanity for the gradients
        d_hat = torch.ones_like(d_i).to(device)
        d_hat[mask_pred] = d_i[mask_pred]
        d_hat[mask_zero_occupied] = 0.

        return d_hat, mask_pred, mask_zero_occupied

    def pixels_to_world(self, pixels, camera_mat, world_mat, scale_mat, c,
                        it=None, sampling_accuracy=None):
        ''' Projects pixels to the world coordinate system.

        Args:
            pixels (tensor): sampled pixels in range [-1, 1]
            camera_mat (tensor): camera matrices
            world_mat (tensor): world matrices
            scale_mat (tensor): scale matrices
            c (tensor): latent conditioned code c
            it (int): training iteration (used for ray sampling scheduler)
            sampling_accuracy (tuple): if not None, this overwrites the default
                sampling accuracy ([128, 129])
        '''
        batch_size, n_points, _ = pixels.shape
        pixels_world = image_points_to_world(pixels, camera_mat, world_mat,
                                             scale_mat)
        camera_world = origin_to_world(n_points, camera_mat, world_mat,
                                       scale_mat)
        ray_vector = (pixels_world - camera_world)

        d_hat, mask_pred, mask_zero_occupied = self.march_along_ray(
            camera_world, ray_vector, c, it, sampling_accuracy)
        p_world_hat = camera_world + ray_vector * d_hat.unsqueeze(-1)
        return p_world_hat, mask_pred, mask_zero_occupied

    def decode_color(self, p_world, c=None, **kwargs):
        ''' Decodes the color values for world points.

        Args:
            p_world (tensor): world point tensor
            c (tensor): latent conditioned code c
        '''
        rgb_hat = self.decoder(p_world, c=c, only_texture=True)
        rgb_hat = torch.sigmoid(rgb_hat)
        return rgb_hat

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            c = torch.empty(inputs.size(0), 0).to(self._device)

        return c

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
