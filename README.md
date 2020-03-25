# Differentiable Volumetric Rendering
<div style="text-align: center">
<img src="media/single-view-reconstruction/ours-combined/airplane1.gif" width="256"/>
<img src="media/single-view-reconstruction/ours-combined/bench.gif" width="256"/>
<img src="media/single-view-reconstruction/ours-combined/car1.gif" width="256"/>
<img src="media/multi-view-reconstruction/birds/shape.gif" width="256"/>
<img src="media/multi-view-reconstruction/birds/normals.gif" width="256"/>
<img src="media/multi-view-reconstruction/birds/texture.gif" width="256"/>
</div>

This repository contains the code for the paper
[Differentiable Volumetric Rendering: Learning Implicit 3D Representations without 3D Supervision](http://www.cvlibs.net/publications/Niemeyer2020CVPR.pdf).

You can find detailed usage instructions for training your own models and using pre-trained models below.

If you find our code or paper useful, please consider citing

    @inproceedings{DVR,
        title = {Differentiable Volumetric Rendering: Learning Implicit 3D Representations without 3D Supervision},
        author = {Niemeyer, Michael and Mescheder, Lars and Oechsle, Michael and Geiger, Andreas},
        booktitle = {Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
        year = {2020}
    }

## Installation

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/).

You can create an anaconda environment called `dvr` using
```
conda env create -f environment.yaml
conda activate dvr
```
Next, compile the extension modules.
You can do this via
```
python setup.py build_ext --inplace
```

## Demo
<div style="text-align: center">
<img src="media/demo/choy_renderings/f32cfab76f75cbebc6ced373b157c8ba_input.jpg" width="250"/> 
<img src="media/single-view-reconstruction/ours-rgb/car2.gif" width="300"/>
<img src="media/single-view-reconstruction/ours-depth/car2.gif" width="300"/>

You can now test our code on the provided input images in the `demo` folder.
To this end, start the generation process for one of the config files in the `configs/demo` folder.
For example, simply run
```
python generate.py configs/demo/demo_combined.yaml
```
This script should create a folder `out/demo/demo_combined` where the output meshes are stored.
The script will copy the inputs into the `generation/inputs` folder and creates the meshes in the `generation/meshes` folder.
Moreover, the script creates a `generation/vis` folder where both inputs and outputs are copied together.

## Dataset

To evaluate a pre-trained model or train a new model from scratch, you have to obtain the respective dataset.
We use three different datasets in the DVR project: 
<ol type="a">
<li>ShapeNet for 2.5D supervised models (using the <a href="https://arxiv.org/abs/1604.00449">Choy et. al. renderings</a> as input and our renderings as supervision) </li>
<li>ShapeNet for 2D supervised models (using the <a href="https://arxiv.org/abs/1711.07566">Kato et. al. renderings</a>)</li>
<li>A subset of the DTU multi-view dataset </li>
</ol>

You can download our preprocessed data using
```
bash scripts/download_data.sh
```
and following the instructions. The sizes of the datasets are 114GB (a), 34GB (b), and 0.5GB (c).

This script should download and unpack the data automatically into the `data` folder.

## Usage
When you have installed all binary dependencies and obtained the preprocessed data, you are ready to run our pre-trained models and train new models from scratch.

### Generation
To generate meshes using a trained model, use
```
python generate.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with the correct config file.

The easiest way is to use a pre-trained model.
You can do this by using one of the config files which are indicated with `_pretrained.yaml`. 

For example, for our 2.5D supervised single-view reconstruction model run
```
python generate.py configs/single_view_reconstruction/multi_view_supervision/ours_depth_pretrained.yaml
```
or for our multi-view reconstruction from RGB images and sparse depth maps for the birds object run
```
python generate.py configs/multi_view_reconstruction/birds/ours_depth_mvs_pretrained.yaml
```

Our script will automatically download the model checkpoints and run the generation.
You can find the outputs in the `out/.../pretrained` folders.

Please note that the config files  `*_pretrained.yaml` are only for generation, not for training new models: when these configs are used for training, the model will be trained from scratch, but during inference our code will still use the pre-trained model.

### Evaluation
For evaluation of the models, we provide the script `eval_meshes.py`. You can run it using
```
python eval_meshes.py CONFIG.yaml
```
The script takes the meshes generated in the previous step and evaluates them using a standardized protocol.
The output will be written to `.pkl`/`.csv` files in the corresponding generation folder which can be processed using [pandas](https://pandas.pydata.org/).

### Training
Finally, to train a new network from scratch, run
```
python train.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with the name of the configuration file you want to use.

You can monitor on <http://localhost:6006> the training process using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
cd OUTPUT_DIR
tensorboard --logdir ./logs
```
where you replace `OUTPUT_DIR` with the respective output directory.

For available training options, please take a look at `configs/default.yaml`.

# Futher Information

## More Work on Implicit Representations
If you like the DVR project, please check out other works on implicit representions from our group:
- [Mescheder et. al. - Occupancy Networks: Learning 3D Reconstruction in Function Space (CVPR 2019)](https://avg.is.tuebingen.mpg.de/publications/occupancy-networks)
- [Oechsle et. al. - Texture Fields: Learning Texture Representations in Function Space (ICCV 2019)](https://avg.is.tuebingen.mpg.de/publications/oechsle2019iccv)
- [Niemeyer et. al. - Occupancy Flow: 4D Reconstruction by Learning Particle Dynamics (ICCV 2019)](https://avg.is.tuebingen.mpg.de/publications/niemeyer2019iccv)

## Other Relevant Works
Also check out other exciting works on inferring implicit representations without 3D supervision: 
- [Liu et. al. - Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning (ICCV 2019)](https://arxiv.org/abs/1904.01786)
- [Sitzmann et. al. - Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations (NeurIPS 2019)](https://arxiv.org/abs/1906.01618)
- [Mildenhall et. al. - NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (ArXiv 2020)](https://arxiv.org/abs/2003.08934)
