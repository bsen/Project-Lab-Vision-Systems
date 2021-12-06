# Project-Lab-Vision-Systems

The final project of the MA-INF 4308 - Lab Vision Systems course at the University of Bonn.

The project is about stereo depth estimation.
That is, we should build a pipeline to estimate the depth of every pixel in an image, using the input of two cameras which are slightly offset.

The Scene Flow dataset (https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) 
was used for pretraining and the KITTI dataset (http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) 
was used for finetuning and evaluation.

## Dependencies
* the KITTI dataset (can be downloaded from http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
	(extract this in a folder called `kitti2015` inside the `datasets` folder of this repository) 
* `pytorch`
* `cuda`
* `torch-warmup-lr` module from Le found under https://github.com/lehduong/torch-warmup-lr

The pretrained models and tensorboard logs can be found on https://uni-bonn.sciebo.de/s/Vlg2afof4GKd7JL/download .
