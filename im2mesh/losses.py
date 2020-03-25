import torch
from torch.nn import functional as F


def apply_reduction(tensor, reduction_method='sum'):
    ''' Applies reduction method to tensor.

    Args:
        tensor (tensor): tensor
        reduction_method (string): reduction method (sum or mean)
    '''
    if reduction_method == 'sum':
        return tensor.sum()
    elif reduction_method == 'mean':
        return tensor.mean()


def l1_loss(val_gt, val_pred, reduction_method='sum', eps=0., sigma_pow=1,
            feat_dim=True):
    ''' Calculates the L1 loss.
    The used formula is (|x - y| + eps)^sigma_pow which reduces to |x - y| for
    eps = 0 and sigma_pow = 1.

    Args:
        val_gt (tensor): GT values
        val_pred (tensor): predicted values
        reduction_method (string): reduction method
        eps (float): epsilon value (see above)
        sigma_pow (float): power value (see above)
        feat_dim (bool): whether the tensors have a feature dimension
    '''
    assert(val_pred.shape == val_gt.shape)
    loss_out = (val_gt - val_pred).abs()
    loss_out = (loss_out + eps).pow(sigma_pow)
    if feat_dim:
        loss_out = loss_out.sum(-1)
    return apply_reduction(loss_out, reduction_method)


def l2_loss(val_gt, val_pred, reduction_method='sum'):
    ''' Applies L2 loss.
    '''
    assert(val_gt.shape == val_pred.shape)
    loss_out = torch.norm((val_gt - val_pred), dim=-1)
    return apply_reduction(loss_out, reduction_method)


def image_gradient_loss(val_pred, val_gt, mask, patch_size,
                        reduction_method='sum'):
    ''' Calculates the L2 loss on the image gradients.
    We assume that tensors have dimensions [B, N, patch_size, patch_size, 3]

    Args:
        val_pred (tensor): predicted values
        val_gt (tensor): GT values
        mask (tensor): which values needs to be masked
        patch_size (int): size of the used patch
        reduction_method (string): reduction method (sum or mean)
    '''
    assert((val_gt.shape == val_pred.shape) &
           (patch_size > 1) &
           (val_gt.shape[1] % (patch_size ** 2) == 0))

    # sanity
    val_gt[mask == 0] = 0.
    rgb_pred = torch.zeros_like(val_pred)
    rgb_pred[mask] = val_pred[mask]
    # Reshape tensors
    batch_size, n_pts, _ = val_gt.shape
    val_gt = val_gt.view(batch_size, -1, patch_size, patch_size, 3)
    rgb_pred = rgb_pred.view(batch_size, -1, patch_size, patch_size, 3)

    # Get mask where all patch entries are valid
    mask_patch = mask.view(
        batch_size, -1, patch_size, patch_size).sum(-1).sum(-1) == \
        patch_size * patch_size

    # Calculate gradients
    ddx = val_gt[:, :, 0, 0] - val_gt[:, :, 0, 1]
    ddy = val_gt[:, :, 0, 0] - val_gt[:, :, 1, 0]
    ddx_pred = rgb_pred[:, :, 0, 0] - rgb_pred[:, :, 0, 1]
    ddy_pred = rgb_pred[:, :, 0, 0] - rgb_pred[:, :, 1, 0]

    # Stack gradient values to 2D tensors
    ddx, ddy = ddx[mask_patch], ddy[mask_patch]
    grad_gt = torch.stack([ddx, ddy], dim=-1)
    ddx_pred, ddy_pred = ddx_pred[mask_patch], ddy_pred[mask_patch]
    grad_pred = torch.stack([ddx_pred, ddy_pred], dim=-1)

    # Calculate l2 norm on 2D vectors
    loss_out = torch.norm(grad_pred - grad_gt, dim=-1).sum(-1)
    return apply_reduction(loss_out, reduction_method)


def cross_entropy_occupancy_loss(logits_pred, is_occupied=True, weights=None,
                                 reduction_method='sum'):
    ''' Calculates the cross entropy occupancy loss.

    Args:
        logits_pred (tensor): predicted logits
        is_occupied (bool): whether the points should be occupied or not
        weights (tensor): whether to weight the points with given tensor
        reduction_method (string): reduction method (sum or mean)
    '''
    if is_occupied:
        occ_gt = torch.ones_like(logits_pred)
    else:
        occ_gt = torch.zeros_like(logits_pred)

    loss_out = F.binary_cross_entropy_with_logits(
        logits_pred, occ_gt, reduction='none')
    if weights is not None:
        assert(loss_out.shape == weights.shape)
        loss_out = loss_out * weights
    return apply_reduction(loss_out, reduction_method)


def occupancy_loss(logits_pred, weights=None, reduction_method='sum'):
    ''' Calculates the occupancy loss.

    Args:
        logits_pred (tensor): predicted logits
        weights (tensor): whether to weight the points with given tensor
        reduction_method (string): reduction method (sum or mean)
    '''
    return cross_entropy_occupancy_loss(logits_pred, weights=weights,
                                        reduction_method=reduction_method)


def freespace_loss(logits_pred, weights=None, reduction_method='sum'):
    ''' Calculates the freespace loss.

    Args:
        logits_pred (tensor): predicted logits
        weights (tensor): whether to weight the points with given tensor
        reduction_method (string): reduction method (sum or mean)
    '''
    return cross_entropy_occupancy_loss(logits_pred, is_occupied=False,
                                        weights=weights,
                                        reduction_method=reduction_method)
