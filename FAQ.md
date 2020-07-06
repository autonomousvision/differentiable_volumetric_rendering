# FAQ

## Camera Matrices

In this project, we use three different matrices to project points from the object-centric coordinate center to the image, and vice versa.
We will discuss the different matrices in the following.

### World Matrix

The [world matrix world_mat](https://github.com/autonomousvision/differentiable_volumetric_rendering/blob/969b17c641107696629e2c739cd023d61e5770f2/im2mesh/data/fields.py#L304) is the extrinsics 4x4 matrix, often referred to as "Rt". When applying this matrix to a point, you transform this point from world space to camera space.

### Camera Matrix

The [camera matrix camera_mat](https://github.com/autonomousvision/differentiable_volumetric_rendering/blob/969b17c641107696629e2c739cd023d61e5770f2/im2mesh/data/fields.py#L305) is the intrinsics 4x4 matrix, often referred to as "K". The only special thing we do is that we define the final pixel coordinates to be in the interval [-1, 1], with [-1, -1] being the top left corner, and the camera is looking at (0, 0, 1). In contrast, the ordinary definition is to have pixel values in [0, W] and [0, H], where (H, W) are the image dimensions, and the camera is sometimes looking at (0, 0, -1). (We use this special definition because we think it fits better the Pytorch framework.)

### Scale Matrix
The [scale matrix scale_mat](https://github.com/autonomousvision/differentiable_volumetric_rendering/blob/969b17c641107696629e2c739cd023d61e5770f2/im2mesh/data/fields.py#L306) is an additional 4x4 matrix for the case when the objects are not centred at the world origin and not in the unit cube. This matrix transforms a 3D point from the "object-centric" space where the object is centred at the origin and in the unit cube to the original world space. In the [data field](https://github.com/autonomousvision/differentiable_volumetric_rendering/blob/969b17c641107696629e2c739cd023d61e5770f2/im2mesh/data/fields.py#L306) one can see that this matrix is just the identity if no other value is given, e.g. for the ShapeNet datasets. We only use this matrix for the DTU dataset, where objects are not centred.

### Transform from the object-centric space to camera space

Using the above matrices, we can easily transform a homogeneous point `p_world` from the object-centric space to camera space via
```
p_cam = camera_mat @ world_mat @ scale_mat @ p_world
```
where `@` indicates matrix multiplication. Please have a look at our [projection function](https://github.com/autonomousvision/differentiable_volumetric_rendering/blob/43194fe8e02349a62bdbb867eefb98ba79bc90eb/im2mesh/common.py#L454) for more details. 

For a more in-depth example, please have a look [this script](https://github.com/autonomousvision/differentiable_volumetric_rendering/blob/master/project_pixels_to_world_example.py).
