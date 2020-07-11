import torch
# import torch.distributions as dist
import os
import shutil
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
import numpy as np
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.common import transform_mesh
from PIL import Image
from scipy.spatial.transform import Rotation as R
import trimesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract meshes from occupancy process.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--upsampling-steps', type=int, default=-1,
                        help='Overrites the default upsampling steps in config')
    parser.add_argument('--refinement-step', type=int, default=-1,
                        help='Overrites the default refinement steps in config')


    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    # Overwrite upsamping and refinement step if desired
    if args.upsampling_steps != -1:
        cfg['generation']['upsampling_steps'] = args.upsampling_steps
    if args.refinement_step != -1:
        cfg['generation']['refinement_step'] = args.refinement_step

    # Shortcuts
    out_dir = cfg['training']['out_dir']
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
    out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
    out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

    batch_size = cfg['generation']['batch_size']
    input_type = cfg['data']['input_type']
    vis_n_outputs = cfg['generation']['vis_n_outputs']
    mesh_extension = cfg['generation']['mesh_extension']

    # Dataset
    # This is for DTU when we parallelise over images
    # we do not want to treat different images from same object as
    # different objects
    cfg['data']['split_model_for_images'] = False
    dataset = config.get_dataset(cfg, mode='test', return_idx=True)

    # Model
    model = config.get_model(cfg, device=device, len_dataset=len(dataset))

    checkpoint_io = CheckpointIO(out_dir, model=model)
    checkpoint_io.load(cfg['test']['model_file'], device=device)
    
    # Generator
    generator = config.get_generator(model, cfg, device=device)

    torch.manual_seed(0)
    # Loader
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=True)

    # Statistics
    time_dicts = []
    vis_file_dict = {}

    # Generate
    model.eval()

    # Count how many models already created
    model_counter = defaultdict(int)
    for it, data in enumerate(tqdm(test_loader)):
        # Output folders
        mesh_dir = os.path.join(generation_dir, 'meshes')
        in_dir = os.path.join(generation_dir, 'input')
        generation_vis_dir = os.path.join(generation_dir, 'vis', )
        generation_paper_vis_dir = os.path.join(
            generation_dir, 'visualizations_paper')

        # Get index etc.
        idx = data['idx'].item()
        try:
            model_dict = dataset.get_model_dict(idx)
        except AttributeError:
            model_dict = {'model': str(idx), 'category': 'na', 'category_id': 0000}

        modelname = model_dict['model']
        category = model_dict.get('category', 'na')
        category_id = model_dict.get('category_id', 0000)
        mesh_dir = os.path.join(mesh_dir, category)
        in_dir = os.path.join(in_dir, category)
        generation_vis_dir = os.path.join(generation_vis_dir, category)
        generation_paper_vis_dir = os.path.join(generation_paper_vis_dir, category)

        if not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)
        if not os.path.exists(generation_vis_dir) and vis_n_outputs > 0:
            os.makedirs(generation_vis_dir)

        # Timing dict
        time_dict = {
            'idx': idx,
            'class_name': category,
            'class_id': category_id,
            'modelname': modelname,
        }
        time_dicts.append(time_dict)

        # Generate outputs
        out_file_dict = {}

        # add empty list to vis_out_file for this category
        if category not in vis_file_dict.keys():
            vis_file_dict[category] = []

        try:
            t0 = time.time()
            out = generator.generate_mesh(data)
            time_dict['mesh'] = time.time() - t0
            # Get statistics
            try:
                mesh, stats_dict = out
            except TypeError:
                mesh, stats_dict = out, {}
            time_dict.update(stats_dict)

            # Write output
            mesh_out_file = os.path.join(
                mesh_dir, '%s.%s' % (modelname, mesh_extension))
            mesh.export(mesh_out_file)
            out_file_dict['mesh'] = mesh_out_file

            # For DTU save also transformed-back mesh to file
            if cfg['data']['dataset_name'] == 'DTU':
                scale_mat = data.get('camera.scale_mat_0')[0]
                mesh_transformed = transform_mesh(mesh, scale_mat)
                mesh_out_file = os.path.join(
                    mesh_dir, '%s_world_scale.%s' % (modelname, mesh_extension))
                mesh_transformed.export(mesh_out_file)

        except RuntimeError:
            print("Error generating mesh %s (%s)." % (modelname, category))

        # Copy to visualization directory for first vis_n_output samples
        c_it = model_counter[category]
        if c_it < vis_n_outputs:
            # add model to vis_out_file
            vis_file_dict[str(category)].append(modelname)
            # Save output files
            for k, filepath in out_file_dict.items():
                ext = os.path.splitext(filepath)[1]
                out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
                                        % (c_it, k, ext))
                if cfg['data']['dataset_name'] == 'DTU':
                    # rotate for DTU to visualization purposes
                    r = R.from_euler('xz', [-90, 10], degrees=True).as_matrix() @ \
                        R.from_euler('xzy', [220, 44.9, 10.6], degrees=True
                                    ).as_matrix()
                    transform = np.eye(4).astype(np.float32)
                    transform[:3, :3] = r.astype(np.float32)
                    mesh = transform_mesh(mesh, transform)
                    mesh.export(out_file)
                else:
                    shutil.copyfile(filepath, out_file)

            if cfg['data']['input_type'] == 'image':
                img = data.get('inputs')[0].permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                out_file = os.path.join(generation_vis_dir, '%02d_input.jpg'
                                        % (c_it))
                img.save(out_file)

        model_counter[category] += 1

    # Create pandas dataframe and save
    time_df = pd.DataFrame(time_dicts)
    time_df.set_index(['idx'], inplace=True)
    time_df.to_pickle(out_time_file)

    # Create pickle files  with main statistics
    time_df_class = time_df.groupby(by=['class_name']).mean()
    time_df_class.to_pickle(out_time_file_class)

    # Print results
    time_df_class.loc['mean'] = time_df_class.mean()
    print('Timings [s]:')
    print(time_df_class)

    # save vis_out_file
    vis_file_dict_name = os.path.join(generation_dir, 'vis', 'visualization_files')
    np.save(vis_file_dict_name, vis_file_dict)
