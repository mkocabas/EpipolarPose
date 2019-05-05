[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-supervised-learning-of-3d-human-pose/3d-human-pose-estimation-human36m)](https://paperswithcode.com/sota/3d-human-pose-estimation-human36m?p=self-supervised-learning-of-3d-human-pose)

# Self-Supervised Learning of 3D Human Pose using Multi-view Geometry (accepted to CVPR2019)

## Introduction
This is a pytorch implementation of 
[*Self-Supervised Learning of 3D Human Pose using Multi-view Geometry*](https://arxiv.org/abs/1903.02330) paper.

> [**Self-Supervised Learning of 3D Human Pose using Multi-view Geometry**](https://arxiv.org/abs/1903.02330),            
> [Muhammed Kocabas](http://user.ceng.metu.edu.tr/~e2270981/)\*, [Salih Karagoz](https://salihkaragoz.github.io/)\*, 
[Emre Akbas](http://user.ceng.metu.edu.tr/~emre/),        
> *IEEE Computer Vision and Pattern Recognition, 2019* (\*equal contribution)   

In this work, we present **_EpipolarPose_**, a self-supervised learning method for 
3D human pose estimation, which does not need any 3D ground-truth data or camera extrinsics.

During training, EpipolarPose estimates 2D poses from multi-view images, and then, utilizes epipolar geometry 
to obtain a 3D pose and camera geometry which are subsequently used to train a 3D pose estimator.

In the test time, it only takes an RGB image to produce a 3D pose result. Check out [`demo.ipynb`](demo.ipynb) to
run a simple demo.

Here we show some sample outputs from our model on the Human3.6M dataset. 
For each set of results we first show the input image, followed by the ground truth, 
fully supervised model and self supervised model outputs.

<p align="center"><img src="https://i.imgur.com/jZph53h.png" width="100%" alt=""/></p>

### Video Demo
<p align="center"><a target=_blank href="http://www.youtube.com/watch?v=lkXBiKRfRDw"><img src="http://img.youtube.com/vi/lkXBiKRfRDw/0.jpg" width="50%" alt="" /></a></p>



## Overview
- `scripts/`: includes training and validation scripts.
- `lib/`: contains data preparation, model definition, and some utility functions.
- `refiner/`: includes the implementation of _refinement unit_ explained in the paper Section 3.3.
- `experiments/`: contains `*.yaml` configuration files to run experiments.
- `sample_images/`: images from Human3.6M dataset to run demo notebook.


## Requirements
The code is developed using python 3.7.1 on Ubuntu 16.04. NVIDIA GPUs ared needed to train and test. 
See [`requirements.txt`](requirements.txt) for other dependencies.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instructions](https://pytorch.org/).
   _Note that if you use pytorch's version < v1.0.0, you should follow the instructions at 
   <https://github.com/Microsoft/human-pose-estimation.pytorch> to disable cudnn's implementation of 
   BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)_
2. Clone this repo, and we will call the directory that you cloned as `${ROOT}`
3. Install dependencies.
   ```
   pip install -r requirements.txt
   ```
4. Download annotation files from [GoogleDrive](https://drive.google.com/open?id=147AlIWRv9QDmp5pGjwG2yMWEG_2-E2ai) 
(150 MB) as a zip file under `${ROOT}` folder. Run below commands to unzip them.
   ```
   unzip data.zip
   rm data.zip
   ```
5. Finally prepare your workspace by running:
    ```bash
    mkdir output
    mkdir models
    ```
    Optionally you can download pretrained weights using the links in the below table. You can put them under `models`
    directory. At the end, your directory tree should like this.

   ```
   ${ROOT}
   ├── data/
   ├── experiments/
   ├── lib/
   ├── models/
   ├── output/
   ├── refiner/
   ├── sample_images/
   ├── scripts/
   ├── demo.ipynb
   ├── README.md
   └── requirements.txt
   ```
6. Yep, you are ready to run [`demo.ipynb`](demo.ipynb).
### Data preparation
You would need Human3.6M data to train or test our model. **For Human3.6M data**, please download from 
[Human 3.6 M dataset](http://vision.imar.ro/human3.6m/description.php). 
You would need to create an account to get download permission. After downloading video files, you can run 
[our script](https://gist.github.com/mkocabas/5669c12debec54b172797743a3c0b778) to extract images.
Then run `ln -s <path_to_extracted_h36m_images> ${ROOT}/data/h36m/images` to create a soft link to images folder.
Currently you can use annotation files we provided in step 4, however we will release the annotation preparation
script soon after cleaning and proper testing.    

If you would like to pretrain an EpipolarPose model on MPII data, 
please download image files from 
[MPII Human Pose Dataset](https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz) (12.9 GB).
Extract it under `${ROOT}/data/mpii` directory. If you already have the MPII dataset, you can create a soft link to images:
`ln -s <path_to_mpii_images> ${ROOT}/data/mpii/images`

During training, we make use of [synthetic-occlusion](https://github.com/isarandi/synthetic-occlusion). If you want to use it please download the Pascal VOC dataset as instructed in their [repo](https://github.com/isarandi/synthetic-occlusion#getting-started) and update the `VOC` parameter in configuration files.

After downloading the datasets your `data` directory tree should look like this: 
```
${ROOT}
|── data/
├───├── mpii/
|   └───├── annot/
|       └── images/
|       
└───├── h36m/
    └───├── annot/
        └── images/
            ├── S1/
            └── S5/
            ...
```
### Pretrained Models

#### Human3.6M
Download pretrained models using the given links, and put them under indicated paths.

| Model                        | Backbone               | MPJPE on Human3.6M (mm) |    Link   | Directory                                 |
|------------------------------|:----------------------:|:-----------------------:|:---------:|------------------------------------------|
| Fully Supervised             | resnet18               | 63.0                    | [model](https://drive.google.com/open?id=1b_1o-gTriiypZlVeZkvoqQfRIqZhpD5T) | `models/h36m/fully_supervised_resnet18.pth.tar`        |
| Fully Supervised             | resnet34               | 59.6                    | [model](https://drive.google.com/open?id=1pDLqOoUjgC-UI979l_u-AV5hJHcFNaOB) | `models/h36m/fully_supervised_resnet34.pth.tar`        |
| Fully Supervised             | resnet50               | 51.8                    | [model](https://drive.google.com/open?id=1TCiUCwyNKviO_-_f6WaOJe3b4pI9uPDh) | `models/h36m/fully_supervised.pth.tar`        |
| Self Supervised R/t          | resnet50               | 76.6                    | [model](https://drive.google.com/open?id=1OVKnQy_TCJXemsAxA9rdEToYVBG4JL4j) | `models/h36m/self_supervised_with_rt.pth.tar` |
| Self Supervised without R/t  | resnet50               | 78.8 (NMPJPE)           | [model](https://drive.google.com/open?id=1iJT0b5vWgdGyseO-ZNMn9S5ATMuutmac) | `models/h36m/self_supervised_wo_rt.pth.tar`   |
| Self Supervised (2D GT)      | resnet50               | 55.0                    | [model](https://drive.google.com/open?id=1RZ32GmikfixiQ6C-CTuccq5qfuxAJj8R) | `models/h36m/self_supervised_2d_gt.pth.tar`   |
| Self Supervised (Subject 1)  | resnet50               | 65.3                    | [model](https://drive.google.com/open?id=1Zo78GY4itUm2fQ85M-v1tYmwoSDkRrW1) | `models/h36m/self_supervised_s1.pth.tar`      |
| Self Supervised + refinement | MLP-baseline           | 60.5                    | [model](https://drive.google.com/open?id=1oW0PYNVd63G1BkRLmYczqqCsUVEsstPJ) | `models/h36m/refiner.pth.tar`                 |

- **Fully Supervised:** trained using ground truth data.
- **Self Supervised R/t:** trained using only camera extrinsic parameters.
- **Self Supervised without R/t:** trained without any ground truth data or camera parameters.
- **Self Supervised (2D GT):** trained with triangulations from ground truth 2D keypoints provided by the dataset.
- **Self Supervised (Subject 1):** trained with only ground truth data of Subject #1.
- **Self Supervised + refinement:** trained with a refinement module. For details of this setting please refer to
[`refiner/README.md`](refiner/README.md)

Check out the [paper](https://arxiv.org/abs/1903.02330) for more details about training strategies of each model.

#### MPII
To train an EpipolarPose model from scratch, you would need the model pretrained on MPII dataset.

| Model         | Backbone| Mean PCK (%) |    Link   | Directory                                 |
|---------------|:------------:|:-----------------------:|:---------:|------------------------------------------|
| MPII Integral | resnet18       | 84.7                    | [model](https://drive.google.com/open?id=1ygdMG5bHcgFSkNCIMJbyDkKhUQebXaoY) | `models/mpii/mpii_integral_r18.pth.tar`|
| MPII Integral | resnet34       | 86.3                    | [model](https://drive.google.com/open?id=1MFzLysHRq8SbO3V6L8o0ZMTKCIsDDGMW) | `models/mpii/mpii_integral_r34.pth.tar`|
| MPII Integral | resnet50       | 88.3                    | [model](https://drive.google.com/open?id=19ee09gyyPOyrAzarQgM_YXqpaCthAfED) | `models/mpii/mpii_integral.pth.tar`|
| MPII heatmap | resnet50       | 88.5                    | [model](https://drive.google.com/open?id=1R4-3uA1o10Gm7zPEemSE2gUBrEF46D1O) | `models/mpii/mpii_heatmap.pth.tar`|


### Validation on H36M using pretrained models
In order to run validation script with a self supervised model, update the `MODEL.RESUME` field of 
[`experiments/h36m/valid-ss.yaml`](experiments/h36m/valid-ss.yaml) with the path to the pretrained weight and run:
```
python scripts/valid.py --cfg experiments/h36m/valid-ss.yaml
```
To run a fully supervised model on validation set, update the `MODEL.RESUME` field of 
[`experiments/h36m/valid.yaml`](experiments/h36m/valid.yaml) with the path to the pretrained weight and run:
```
python scripts/valid.py --cfg experiments/h36m/valid.yaml
```

### Training on H36M
To train a self supervised model, try:
```
python scripts/train.py --cfg experiments/h36m/train-ss.yaml
```
Fully supervised model:
```
python scripts/train.py --cfg experiments/h36m/train.yaml
```
### Citation
If this work is useful for your research, please cite our [paper](https://arxiv.org/abs/1903.02330):
```
@inproceedings{kocabas2019epipolar,
    author = {Kocabas, Muhammed and Karagoz, Salih and Akbas, Emre},
    title = {Self-Supervised Learning of 3D Human Pose using Multi-view Geometry},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
}
```
### References

- [Integral Human Pose](https://github.com/JimmySuen/integral-human-pose)
- [Simple Baselines for Human Pose Estimation](https://github.com/Microsoft/human-pose-estimation.pytorch/)  
- [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline)
- [Synthetic-occlusion](https://github.com/isarandi/synthetic-occlusion)

We thank to the authors for releasing their codes. Please also consider citing their works.

### License
This code is freely available for free non-commercial use, and may be redistributed under these conditions. 
Please, see the [LICENSE](LICENSE) for further details. 
Third-party datasets and softwares are subject to their respective licenses. 
