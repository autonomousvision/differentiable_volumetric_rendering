from torch import nn
import os
from im2mesh.encoder import encoder_dict
from im2mesh.dvr import models, training, generation, rendering
from im2mesh import data


def get_model(cfg, device=None, len_dataset=0, **kwargs):
    ''' Returns the DVR model.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
        len_dataset (int): length of dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    depth_function_kwargs = cfg['model']['depth_function_kwargs']
    # Add the depth range to depth function kwargs
    depth_range = cfg['data']['depth_range']
    depth_function_kwargs['depth_range'] = depth_range

    # Load always the decoder
    decoder = models.decoder_dict[decoder](
        dim=dim, c_dim=c_dim, **decoder_kwargs
    )

    # Load encoder
    if encoder == 'idx':
        encoder = nn.Embedding(len_dataset, c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](c_dim=c_dim, **encoder_kwargs)
    else:
        encoder = None

    model = models.DVR(
        decoder, encoder=encoder, device=device,
        depth_function_kwargs=depth_function_kwargs,
    )
    return model


def get_trainer(model, optimizer, cfg, device, generator, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the DVR model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
        generator (Generator): generator instance to 
            generate meshes for visualization
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    n_training_points = cfg['training']['n_training_points']
    lambda_freespace = cfg['model']['lambda_freespace']
    lambda_occupied = cfg['model']['lambda_occupied']
    lambda_rgb = cfg['model']['lambda_rgb']
    n_eval_points = cfg['training']['n_eval_points']
    lambda_depth = cfg['model']['lambda_depth']
    lambda_image_gradients = cfg['model']['lambda_image_gradients']
    patch_size = cfg['model']['patch_size']
    reduction_method = cfg['model']['reduction_method']
    sample_continuous = cfg['training']['sample_continuous']
    lambda_sparse_depth = cfg['model']['lambda_sparse_depth']
    overwrite_visualization = cfg['training']['overwrite_visualization']
    depth_from_visual_hull = cfg['data']['depth_from_visual_hull']
    depth_range = cfg['data']['depth_range']
    depth_loss_on_world_points = cfg['training']['depth_loss_on_world_points']
    occupancy_random_normal = cfg['training']['occupancy_random_normal']
    use_cube_intersection = cfg['training']['use_cube_intersection']
    always_freespace = cfg['training']['always_freespace']
    multi_gpu = cfg['training']['multi_gpu']
    lambda_normal = cfg['model']['lambda_normal']

    trainer = training.Trainer(
        model, optimizer, device=device, vis_dir=vis_dir, threshold=threshold,
        n_training_points=n_training_points, 
        lambda_freespace=lambda_freespace, lambda_occupied=lambda_occupied,
        lambda_rgb=lambda_rgb, lambda_depth=lambda_depth, generator=generator,
        n_eval_points=n_eval_points,
        lambda_image_gradients=lambda_image_gradients,
        patch_size=patch_size, reduction_method=reduction_method,
        sample_continuous=sample_continuous,
        lambda_sparse_depth=lambda_sparse_depth,
        overwrite_visualization=overwrite_visualization,
        depth_from_visual_hull=depth_from_visual_hull,
        depth_range=depth_range,
        depth_loss_on_world_points=depth_loss_on_world_points,
        occupancy_random_normal=occupancy_random_normal,
        use_cube_intersection=use_cube_intersection,
        always_freespace=always_freespace, multi_gpu=multi_gpu,
        lambda_normal=lambda_normal)

    return trainer


def get_renderer(model, cfg, device, **kwargs):
    ''' Returns the renderer object.

    Args:
        model (nn.Module): DVR model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''

    renderer = rendering.Renderer(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        colors=cfg['rendering']['colors'],
        resolution=cfg['rendering']['resolution'],
        n_views=cfg['rendering']['n_views'],
        extension=cfg['rendering']['extension'],
        background=cfg['rendering']['background'],
        ray_sampling_accuracy=cfg['rendering']['ray_sampling_accuracy'],
        n_start_view=cfg['rendering']['n_start_view'],
    )
    return renderer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): DVR model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        padding=cfg['generation']['padding'],
        with_color=cfg['generation']['with_colors'],
        refine_max_faces=cfg['generation']['refine_max_faces'],
    )
    return generator


def get_data_fields(cfg, mode='train'):
    ''' Returns the data fields.

    Args:
        cfg (dict): imported yaml config
        mode (str): the mode which is used
    '''
    resize_img_transform = data.ResizeImage(cfg['data']['img_size'])
    all_images = mode == 'render'
    with_depth = (cfg['model']['lambda_depth'] != 0)
    depth_from_visual_hull = cfg['data']['depth_from_visual_hull']
    random_view = True if (
        mode == 'train' or
        ((cfg['data']['dataset_name'] == 'NMR') and mode == 'test') or
        ((cfg['data']['dataset_name'] == 'NMR') and mode == 'val')
    ) else False

    fields = {}
    if mode in ('train', 'val', 'render'):
        img_field = data.ImagesField(
            cfg['data']['img_folder'], cfg['data']['mask_folder'],
            cfg['data']['depth_folder'],
            transform=resize_img_transform,
            extension=cfg['data']['img_extension'],
            mask_extension=cfg['data']['mask_extension'],
            depth_extension=cfg['data']['depth_extension'],
            with_camera=cfg['data']['img_with_camera'],
            with_mask=cfg['data']['img_with_mask'],
            with_depth=with_depth,
            random_view=random_view,
            dataset_name=cfg['data']['dataset_name'],
            all_images=all_images,
            n_views=cfg['data']['n_views'],
            depth_from_visual_hull=depth_from_visual_hull,
            visual_hull_depth_folder=cfg['data']['visual_hull_depth_folder'],
            ignore_image_idx=cfg['data']['ignore_image_idx'],
        )
        fields['img'] = img_field

        if cfg['model']['lambda_sparse_depth'] != 0:
            fields['sparse_depth'] = data.SparsePointCloud(
                ignore_image_idx=cfg['data']['ignore_image_idx'],
            )

    elif cfg['data']['dataset_name'] == 'DTU':
        fields['camera'] = data.CameraField(
            cfg['data']['n_views'],
        )

    return fields
