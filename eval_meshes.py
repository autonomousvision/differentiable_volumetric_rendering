import argparse
# import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import trimesh
import torch
from im2mesh import config, data
from im2mesh.eval import MeshEvaluator


parser = argparse.ArgumentParser(
    description='Evaluate mesh algorithms.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
mesh_extension = cfg['generation']['mesh_extension']

# Shorthands
out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
eval_dir = os.path.join(out_dir, 'eval')
eval_file_name = cfg['test']['eval_file_name']
out_file = os.path.join(generation_dir, eval_file_name + 'full.pkl')
out_file_class = os.path.join(generation_dir, eval_file_name + '.csv')

pointcloud_field = data.PointCloudField(
    cfg['data']['pointcloud_chamfer_file']
)
fields = {
    'pointcloud_chamfer': pointcloud_field,
    'idx': data.IndexField(),
}

test_split = cfg['data']['test_split']
print('Test split: ', cfg['data']['test_split'])

dataset_folder = cfg['data']['path']
dataset = data.Shapes3dDataset(
    dataset_folder, fields,
    cfg['data']['test_split'],
    categories=cfg['data']['classes'])

# Evaluator
evaluator = MeshEvaluator(n_points=100000)

# Loader
torch.manual_seed(0)
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=True)

if True and not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

# Evaluate all classes
eval_dicts = []
print('Evaluating meshes...')
for it, data in enumerate(tqdm(test_loader)):
    if data is None:
        print('Invalid data.')
        continue

    # Output folders
    mesh_dir = os.path.join(generation_dir, 'meshes')
    # pointcloud_dir = os.path.join(generation_dir, 'pointcloud')

    # Get index etc.
    idx = data['idx'].item()

    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}

    modelname = model_dict['model']
    category_id = model_dict['category']

    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'

    if category_id != 'n/a':
        mesh_dir = os.path.join(mesh_dir, category_id)
        # pointcloud_dir = os.path.join(pointcloud_dir, category_id)

    # Evaluate
    pointcloud_tgt = data['pointcloud_chamfer'].squeeze(0).numpy()
    normals_tgt = data['pointcloud_chamfer.normals'].squeeze(0).numpy()

    # Evaluating mesh
    # Start row and put basic information inside
    eval_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
    }
    eval_dicts.append(eval_dict)

    # Evaluate mesh
    mesh_file = os.path.join(mesh_dir, '%s.%s' % (modelname, mesh_extension))

    if os.path.exists(mesh_file):
        try:
            mesh = trimesh.load(mesh_file, process=False)
            eval_dict_mesh = evaluator.eval_mesh(
                mesh, pointcloud_tgt, normals_tgt)
            for k, v in eval_dict_mesh.items():
                eval_dict[k + ' (mesh)'] = v
        except Exception as e:
            print('Warning: mesh cannot be loaded: %s (%s)' % (mesh_file, e))
    else:
        print('Warning: mesh does not exist: %s' % mesh_file)

# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)
eval_df.to_pickle(out_file)

# Create CSV file  with main statistics
eval_df_class = eval_df.groupby(by=['class name']).mean()
eval_df_class.to_csv(out_file_class)

# Print results
eval_df_class.loc['mean'] = eval_df_class.mean()
print(eval_df_class)
