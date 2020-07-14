import torch
# import torch.distributions as dist
import os
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from PIL import Image
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract meshes from occupancy process.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    # fix
    cfg['data']['split_model_for_images'] = False
    cfg['data']['depth_from_visual_hull'] = False


    out_dir = cfg['training']['out_dir']
    render_dir = os.path.join(out_dir, cfg['rendering']['render_dir'])
    out_time_file = os.path.join(render_dir, 'time_generation_full.pkl')
    out_time_file_class = os.path.join(render_dir, 'time_generation.pkl')
    vis_n_outputs = cfg['generation']['vis_n_outputs']
    input_type = cfg['data']['input_type']

    # Dataset
    dataset = config.get_dataset(cfg, mode='render', return_idx=True)

    # Model
    model = config.get_model(cfg, device=device, len_dataset=len(dataset))

    checkpoint_io = CheckpointIO(out_dir, model=model)
    checkpoint_io.load(cfg['test']['model_file'], device=device)

    # Generator
    renderer = config.get_renderer(model, cfg, device=device)

    # Loader
    torch.manual_seed(0)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=True)

    # Statistics
    time_dicts = []

    # Generate
    model.eval()

    # Count how many models already created
    model_counter = defaultdict(int)

    for it, data in enumerate(tqdm(test_loader)):
        # Output folders
        img_dir = os.path.join(render_dir)
        generation_vis_dir = os.path.join(img_dir, 'vis', )
        if not os.path.exists(generation_vis_dir):
            os.makedirs(generation_vis_dir)

        # Get index etc.
        idx = data['idx'].item()
        try:
            model_dict = dataset.get_model_dict(idx)
        except AttributeError:
            model_dict = {'model': str(idx), 'category': 'na', 'category_id': 0000}

        modelname = model_dict['model']
        category = model_dict.get('category', 'na')
        category_id = model_dict.get('category_id', 0000)

        img_dir = os.path.join(img_dir, category)
        generation_vis_dir = os.path.join(generation_vis_dir, category)

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
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

        t0 = time.time()
        out = renderer.render_and_export(data, img_dir, modelname)
        time_dict['rendering'] = time.time() - t0

        # Get statistics
        try:
            img_in, img_out, stats_dict = out
        except TypeError:
            img_in, img_out, stats_dict = out, {}
        time_dict.update(stats_dict)
        # Copy to visualization directory for first vis_n_output samples
        c_it = model_counter[category]
        if c_it < vis_n_outputs:
            # Save output files
            img_gt = Image.fromarray(
                (img_in[0].permute(1, 2, 0) * 255).numpy().astype(np.uint8)
                ).convert("RGB")
            img_gt.save(
                os.path.join(generation_vis_dir, '%02d_input.png' % c_it)
            )
            img_out[0].save(
                os.path.join(generation_vis_dir, '%02d_pred.png' % c_it)

            )

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
