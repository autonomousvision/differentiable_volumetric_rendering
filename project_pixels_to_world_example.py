import os
import numpy as np 
from PIL import Image 
import imageio
import torch
import trimesh
from im2mesh.common import (get_tensor_values,
    transform_to_world, sample_patch_points
)

'''
This script is an example how pixels with corresponding depth values can be projected to the object-centric world coordinate. 
Given the variable PATH, it assumes folder with images, masks, and depth maps, respectively (see below). 
Then, n_sample_points are sampled from each image and, if the depth value is not inf, is projected to the object-centric world space.
It loops over the first n_images. The final point cloud is then saved to a ply file in out/pcd.ply.
'''


PATH = 'data/DTU/scan65/scan65'
img_folder = 'image'
depth_folder = 'depth'
mask_folder = 'mask'
n_sample_points = 1000
n_images = 49

img_path = os.path.join(PATH, img_folder)
depth_path = os.path.join(PATH, depth_folder)

img_list = [os.path.join(img_path, i) for i in os.listdir(img_path)]
img_list.sort()
depth_list = [os.path.join(depth_path, i) for i in os.listdir(depth_path)]
depth_list.sort()

cam = np.load(os.path.join(PATH, 'cameras.npz'))

v_out = []
rgb_out = []

for i, imgp in enumerate(img_list):
    # load files
    img = np.array(Image.open(imgp).convert("RGB")).astype(np.float32) / 255
    h, w, c = img.shape
    depth = np.array(imageio.imread(depth_list[i]))
    depth = depth.reshape(depth.shape[0], depth.shape[1], -1)[..., 0]

    hd, wd = depth.shape 
    assert(h == hd and w == wd)
    
    p = sample_patch_points(1, n_sample_points, patch_size=1, image_resolution=(h, w), continuous=False)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) 
    depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(1)
    rgb = get_tensor_values(img, p)
    d = get_tensor_values(depth, p)
    mask = (d != np.inf).squeeze(-1)
    d = d[mask].unsqueeze(0)
    rgb = rgb[mask]
    p = p[mask].unsqueeze(0)
 
    # transform to world
    cm = cam.get('camera_mat_%d' % i).astype(np.float32).reshape(1, 4, 4)
    wm = cam.get('world_mat_%d' % i).astype(np.float32).reshape(1, 4, 4)
    sm = cam.get('scale_mat_%d' % i, np.eye(4)).astype(np.float32).reshape(1, 4, 4)
    p_world = transform_to_world(p, d, cm, wm, sm)[0]
    v_out.append(p_world)

v = np.concatenate(v_out, axis=0)
mesh = trimesh.Trimesh(vertices=v).export('out/pcd.ply')
