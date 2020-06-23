import torch
import os
import argparse
from tqdm import tqdm, trange
import numpy as np
from im2mesh import config
import open3d 
from skimage.morphology import binary_dilation, disk

if __name__ == '__main__':
    # Adjust this to your paths; the input path should point to the 
    # DTU dataset including the mvs data which can be downloaded here
    # http://roboimagedata.compute.dtu.dk/
    INPUT_PATH = '/your/dtu/path'
    INPUT_PATH = os.path.join(INPUT_PATH, 'MVS_Data', 'Points')
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("The input path is not pointing to the DTU Dataset. " + \
            "Please download the DTU Dataset and adjust your input path.")

    methods = ['furu', 'tola', 'camp']
    parser = argparse.ArgumentParser(
        description='Filter the DTU baseline predictions with the object masks.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")


    def filter_points(p, data):
        n_images = data.get('img.n_images').item()
        p = torch.from_numpy(p)
        n_p = p.shape[0]
        inside_mask = np.ones((n_p,), dtype=np.bool)
        inside_img = np.zeros((n_p,), dtype=np.bool)
        for i in trange(n_images):
            # get data
            datai = data.get('img.img%i' % i)
            maski_in = datai.get('mask')[0]

            # Apply binary dilation to account for errors in the mask
            maski = torch.from_numpy(binary_dilation(maski_in, disk(12))).float()

            #h, w = maski.shape
            h, w = maski.shape
            w_mat = datai.get('world_mat')[0]
            c_mat = datai.get('camera_mat')[0]
            s_mat = datai.get('scale_mat')[0]

            # project points into image
            phom = torch.cat([p, torch.ones(n_p, 1)], dim=-1).transpose(1, 0)
            proj = c_mat @ w_mat @ phom
            proj = (proj[:2] / proj[-2].unsqueeze(0)).transpose(1, 0)

            # check which points are inside image; by our definition,
            # the coordinates have to be in [-1, 1]
            mask_p_inside = ((proj[:, 0] >= -1) &
                (proj[:, 1] >= -1) &
                (proj[:, 0] <= 1) &
                (proj[:, 1] <= 1)
            )
            inside_img |= mask_p_inside.cpu().numpy()

            # get image coordinates
            proj[:, 0] = (proj[:, 0] + 1) * (w - 1) / 2.
            proj[:, 1] = (proj[:, 1] + 1) * (h - 1) / 2.
            proj = proj.long()

            # fill occupancy values
            proj = proj[mask_p_inside]
            occ = torch.ones(n_p)
            occ[mask_p_inside] = maski[proj[:, 1], proj[:, 0]]
            inside_mask &= (occ.cpu().numpy() >= 0.5)

        occ_out = np.zeros((n_p,))
        occ_out[inside_img & inside_mask] = 1.

        return occ_out

    # Shortcuts
    out_dir = cfg['training']['out_dir']
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])

    if not os.path.isdir(generation_dir):
        os.makedirs(generation_dir)

    # Dataset
    dataset = config.get_dataset(cfg, mode='render', return_idx=True)

    # Loader
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=True)

    for it, data in enumerate(tqdm(test_loader)):
        idx = data['idx'].item()
        try:
            model_dict = dataset.get_model_dict(idx)
        except AttributeError:
            model_dict = {'model': str(idx), 'category': 'na', 'category_id': 0000}

        modelname = model_dict['model']
        category = model_dict.get('category', 'na')
        category_id = model_dict.get('category_id', 0000)
        scan_id = int(modelname[4:])


        for method in methods:
            out_dir = os.path.join(generation_dir, method)

            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            
            in_dir = os.path.join(INPUT_PATH, method)
            scan_path = os.path.join(in_dir, '%s%03d_l3.ply' % (method, scan_id))
            out_file = os.path.join(out_dir, '%s.ply' % modelname)
            if not os.path.exists(out_file):
                pcl = open3d.read_point_cloud(scan_path)
                p = np.asarray(pcl.points).astype(np.float32)
                occ = filter_points(p, data) > 0.5
                pcl.points = open3d.Vector3dVector(p[occ])
                if len(pcl.colors) != 0:
                    c = np.asarray(pcl.colors)
                    pcl.colors = open3d.Vector3dVector(c[occ])
                if len(pcl.normals) != 0:
                    n = np.asarray(pcl.normals)
                    pcl.normals = open3d.Vector3dVector(n[occ])
                open3d.write_point_cloud(out_file, pcl)
