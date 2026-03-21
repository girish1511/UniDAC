
<div align="center">
<h1>UniDAC: Universal Metric Depth Estimation for Any Camera</h1>

[**Girish Chandar Ganesan**](https://girish1511.github.io/)<sup>1</sup> . [**Yuliang Guo**](https://yuliangguo.github.io/)<sup>2</sup> · [**Liu Ren**](https://sites.google.com/site/liurenshomepage/)<sup>2</sup> . [**Xiaoming Liu**](https://scholar.google.com/citations?hl=en&user=Bii0w1oAAAAJ)<sup>1</sup>

<sup>1</sup>Michigan State University&emsp;&emsp;&emsp;&emsp;<sup>2</sup>Bosch Research North America


<a href=''><img src='https://img.shields.io/badge/arXiv-UniDAC-red' alt='Paper PDF'></a>
<a href=''><img src='https://img.shields.io/badge/Project_Page-UniDAC-green' alt='Project Page'></a>
<a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow'></a>

[**CVPR 2026**](https://cvpr.thecvf.com/Conferences/2026)

</div>


<!-- <p align="center">
  <img src="docs/teaser.png" alt="teaser" style="width: 70%;">
</p>

**Depth Any Camera (DAC)** is a zero-shot metric depth estimation framework that extends a perspective-trained model to handle any type of camera with varying FoVs effectively. 

Notably, DAC can be **trained exclusively on perspective images**, yet it generalizes seamlessly to **fisheye** and **360** cameras without requiring specialized training data. Key features include:

1. **Zero-shot metric** depth estimation on **fisheye** and **360** images, significantly outperforming prior metric depth SoTA [Metric3D-v2](https://jugghm.github.io/Metric3Dv2/) and [UniDepth](https://github.com/lpiccinelli-eth/UniDepth/tree/main).
2. Geometry-based training framework **adaptable to any network architecture**, extendable to **other 3D perception tasks**.

**Tired of collecting new data for specific cameras?** DAC maximizes the utility of every existing 3D data for training, regardless of the specific camera types used in new applications. -->

## News

- ``2026-03-21``: Demo code for easy setup and usage.
- ``2026-03-13``: Release of pre-trained UniDAC models trained on moderately sized datasets.
- ``2026-03-13``: Testing and evaluation pipeline for zero-shot metric depth estimation on perspective, fisheye, and 360-degree datasets.
- ``2026-03-13``: Complete UniDAC training pipeline using mixed perspective camera data.
- ``2026-03-13``: Complete data preparation and curation scripts.
- ``2026-02-20``: UniDAC accepted by CVPR 2026!
- [TBD] Foundation-level model trained on a large-scale, diverse dataset mixture, encompassing perspective, fisheye, and 360-degree camera data.

## Visualization

### ScanNet++ fisheye

<!-- The zero-shot metric depth estimation results of Depth Any Camera (DAC) are visualized on ScanNet++ fisheye videos and compared to Metric3D-v2. The visualizations of A.Rel error against ground truth highlight the superior performance of DAC. 


<p align="center">
  <img src="docs/video_scannet++_1.gif" alt="animated" />
</p> -->

### Matterport3D single-view reconstruction

<!-- Additionally, we showcase DAC's application on 360-degree images, where a single forward pass of depth estimation enables full 3D scene reconstruction.

<p align="center">
  <img src="docs/video_matterport3d_1.gif" alt="animated" />
</p>

Additional visual results and comparison with the prior SoTA can be found at <a href='https://yuliangguo.github.io/depth-any-camera/'><img src='https://img.shields.io/badge/Project_Page-Depth Any Camera-green' alt='Project Page'></a> -->




## Pipeline

![pipeline](docs/pipeline.png)

## Performance

UniDAC outperforms all prior metric depth estimation methods trained with perspective images on both indoor and outdoor datasets and sets the SoTA in zero-shot cross-camera generalization and universal domain robustness.
UniDAC outperforms UniK3D, even though the latter has been trained on large FoV images and has a much larger training set, demonstrating the robustness of UniDAC.
Matterport3D is present in the training set of UniK3D and thus we omit its results.

<table cellspacing="0" cellpadding="8">
<thead>
<tr>
<th rowspan="2">Methods</th>
<th rowspan="2">Dataset<br>Size</th>
<th colspan="2">ScanNet++</th>
<th colspan="2">Pano3D-GV2</th>
<th colspan="2">KITTI-360</th>
<th colspan="2">Matterport3D</th>
</tr>
<tr>
<th>δ₁ ↑</th><th>Abs.Rel ↓</th>
<th>δ₁ ↑</th><th>Abs.Rel ↓</th>
<th>δ₁ ↑</th><th>Abs.Rel ↓</th>
<th>δ₁ ↑</th><th>Abs.Rel ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td style="border-bottom:1px solid black;">UniK3D</td>
<td style="border-bottom:1px solid black;">8M</td>
<td style="border-bottom:1px solid black;">0.651</td>
<td style="border-bottom:1px solid black;">0.253</td>
<td style="border-bottom:1px solid black;"><b>0.785</b></td>
<td style="border-bottom:1px solid black;">0.170</td>
<td style="border-bottom:1px solid black;">0.817</td>
<td style="border-bottom:1px solid black;">0.244</td>
<td style="border-bottom:1px solid black;">-</td>
<td style="border-bottom:1px solid black;">-</td>
</tr>
<tr>
<td>Metric3Dv2</td>
<td>16M</td>
<td>0.536</td><td>0.223</td>
<td>0.404</td><td>0.307</td>
<td>0.716</td><td>0.200</td>
<td>0.438</td><td>0.292</td>
</tr>
<tr>
<td>UniDepth</td>
<td>3M</td>
<td>0.364</td><td>0.497</td>
<td>0.247</td><td>0.789</td>
<td>0.481</td><td>0.294</td>
<td>0.258</td><td>0.765</td>
</tr>
<tr>
<td>DAC<sub>U</sub></td>
<td>0.8M</td>
<td>0.658</td><td>0.233</td>
<td>0.684</td><td>0.203</td>
<td>0.708</td><td>0.186</td>
<td>0.662</td><td>0.215</td>
</tr>
<tr>
<td><b>UniDAC</b></td>
<td>1.45M</td>
<td><b>0.918</b></td><td><b>0.097</b></td>
<td>0.768</td><td><b>0.161</b></td>
<td><b>0.836</b></td><td><b>0.141</b></td>
<td><b>0.745</b></td><td><b>0.175</b></td>
</tr>
</tbody>
</table>



## Installation
### Clone the Repository

```bash
git clone https://github.com/girish1511/UniDAC
cd UniDAC
```

<!-- ### Docker Installation 
This repository can be run from within Docker, as long as the NVIDIA Container Toolkit is properly configured.
For Ubuntu Installation steps, refer to [this guide](https://github.com/garylvov/dev_env/tree/main/setup_scripts/nvidia).

```bash
# Build the container
docker build -t dac:latest .
# Enter the container
docker run --gpus all --network host -v $(pwd):/depth_any_camera --rm -it dac /bin/bash 

# Once within the container, 
#  source the post-entry-hooks.sh to finish the install.
source post-entry-hooks.sh
``` -->

### Conda Installation
Alternatively, this repository can be run from within Conda alone.
```bash
conda create -n unidac python=3.10.18 -y
conda activate unidac
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Data Preparation

The training set consist of 4 outdoor datasets and 3 indoor datasets. The testing set consists of two 360 datasets, two fisheye datasets and 4 perspective datasets.

Please refer to [DATA.md](docs/DATA.md) for detailed datasets preparation.

<!-- Our current training set is very slim compared to prior fundation models. Currently, DAC is trained on a combination set of 3 labeled datasets (670k images) for indoor model and a combination of 2 datasets (130k) for outdoor model. Two 360 datasets and two fisheye datasets are used for zero-shot testing.

![data](docs/table_data_coverage.png)

Please refer to [DATA.md](docs/DATA.md) for detailed datasets preparation. Make sure the relative paths of datasets have been set correctly before proceeding to the actual testing and training sections. -->


## Pre-trained models

<!-- We provide two indoor models and two outdoor modeling considering Resnet101 and SwinTransformer-Large (SwinL) as backbones. In addition, we also provide two weaker baseline models for comparison. The download links can be found in the following table or from <a href='https://huggingface.co/yuliangguo/depth-any-camera'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow'></a>. We suggest to save download both the **model configs and model weights** at `checkpoints` in order to run our scripts directly.

| Model Name | Training Datasets | Model Configs | Weights
|:-|:-:|:-:|:-:|
| dac-indoor-resnet101 (ours) | indoor mix 670k | [huggingface](https://huggingface.co/yuliangguo/depth-any-camera/blob/main/dac_resnet101_indoor.json)   | [huggingface](https://huggingface.co/yuliangguo/depth-any-camera/blob/main/dac_resnet101_indoor.pt)|
| dac-indoor-swinL (ours)     | indoor mix 670k | [huggingface](https://huggingface.co/yuliangguo/depth-any-camera/blob/main/dac_swinl_indoor.json) | [huggingface](https://huggingface.co/yuliangguo/depth-any-camera/blob/main/dac_swinl_indoor.pt)|
| dac-outdoor-resnet101 (ours) | outdoor mix 130k | [huggingface](https://huggingface.co/yuliangguo/depth-any-camera/blob/main/dac_resnet101_outdoor.json) | [huggingface](https://huggingface.co/yuliangguo/depth-any-camera/blob/main/dac_resnet101_outdoor.pt)|
| dac-outdoor-swinL (ours)    | outdoor mix 130k | [huggingface](https://huggingface.co/yuliangguo/depth-any-camera/blob/main/dac_swinl_outdoor.json) | [huggingface](https://huggingface.co/yuliangguo/depth-any-camera/blob/main/dac_swinl_outdoor.pt)|
| idisc-metric3d-indoor-resnet101 (weak baseline 1) | indoor mix 670k | [huggingface](https://huggingface.co/yuliangguo/depth-any-camera/blob/main/idisc_resnet101_indoor.json)   | [huggingface](https://huggingface.co/yuliangguo/depth-any-camera/blob/main/idisc_resnet101_indoor.pt)|
| cnndepth-metric3d-indoor-resnet101 (weak baseline 2) | indoor mix 670k | [huggingface](https://huggingface.co/yuliangguo/depth-any-camera/blob/main/cnndepth_resnet101_indoor.json)   | [huggingface](https://huggingface.co/yuliangguo/depth-any-camera/blob/main/cnndepth_resnet101_indoor.pt)| -->

## Demo

<!-- We have provided a ready-to-run demo scripts in the `demo` folder. `demo/demo_dac_indoor.py` demonstrates how to perform inference on various types of camera data, including ScanNet++ (fisheye), Matterport3D (360), and NYU (perspective), using a single metric depth model trained on perspective images. The code generates point cloud files `*.ply` and visualization results as shown below:

<p align="center">
  <img src="demo/output/scannetpp_output_remap_subplot.jpg" alt="demo output" style="width: 70%;">
</p>

`demo/demo_dac_outdoor.py` similarly demonstrates how a single outdoor model handle different types of camera data, including kitti (perspective) and kitti360 (fisheye).

Instead, we also provide demo script for dealing one sample, you may follow the following example command:

```bash
python demo/demo_dac_single.py --config-file checkpoints/dac_swinl_indoor.json --model-file checkpoints/dac_swinl_indoor.pt --sample-file demo/input/scannetpp_sample.json --out-dir demo/output
```

Our depth results can be directly downloaded from [ZipNeRF DAC results](https://huggingface.co/yuliangguo/depth-any-camera/blob/main/zipnerf_dac_swinl_indoor_2025_01.zip), and [ScanNet++ DAC results](https://huggingface.co/yuliangguo/depth-any-camera/blob/main/scannetpp_dac_swinl_indoor_2025_01.zip). The results folder can be simply merged with the original dataset for further usage. The predicted depth maps are saved in uint16 images, with visualization as shown below. 

<p align="center">
  <img src="docs/zipnerf_alameda_vis.gif" alt="animated" style="width: 85%;"/>
</p>

**Important note**: Our depth maps record the Euclidean distance to the camera center instead of the z-distance to the image plane, which is not appropriate for large-FoV camera models. -->

## Testing

<!-- Given provided pretrained models saved in `checkpoints/`, the following code can be run to test and evaluate on certain dataset, e.g., ScanNet++:

```bash
python script/test_dac.py --model-file checkpoints/dac_swinl_indoor.pt --model-name IDiscERP --config-file configs/test/dac_swinl_indoor_test_scannetpp.json --base-path datasets --vis
```

Different config files for testing all the reported datasets are included in [configs/test](configs/test). Interested users could also refer to the provided [lauch.json](.vscode/launch.json) for convinient use or debug provided testing scripts in VSCode. The following tables lay out those most relative ones.

| Testing dataset | Testing script | --model-file | --config-file | --model-name |
|:-|:-|:-|:-|:-:|
| Matterport | scripts/test_dac.py           | checkpoints/dac-indoor-resnet101.pt          | [relative path](configs/test/dac_resnet101_indoor_test_m3d.json)       | IDiscERP |
| Gibson-V2 | ^                              | ^                                            | [relative path](configs/test/dac_resnet101_indoor_test_gv2.json)       | IDiscERP |
| ScanNet++ | ^                              | ^                                            | [relative path](configs/test/dac_resnet101_indoor_test_scannetpp.json) | IDiscERP |
| NYU     | ^                                | ^                                            | [relative path](configs/test/dac_resnet101_indoor_test_nyu.json)       | IDiscERP |
| KITTI360 | ^                               | checkpoints/dac-outdoor-resnet101.pt         | [relative path](configs/test/dac_resnet101_indoor_test_kitti360.json)  | IDisc    |
| KITTI   | ^                                | ^                                            | [relative path](configs/test/dac_resnet101_indoor_test_kitti.json)     | IDisc    |
| ...     | scripts/test_persp.py            | checkpoints/idisc-...                        | ...                                                                    | IDisc    |
| ...     | ^                                | checkpoints/cnndepth-...                     | ...                                                                    | CNNDepth |

**Note**: *IDiscERP* is our modified version of the *IDisc* model, incorporating isolated image and positional encoding features. It has been observed to improve results in small-size data training, particularly for better depth-scale equivariance. However, these modifications are not essential for large dataset training. *CNNDepth* refers to the CNN portion of the *IDisc* model, which serves as a network baseline but consistently underperforms compared to other models.

The ResNet101 models and configuration files can be replaced with the corresponding Swin-L versions. Ensure that the `--model-name` parameter matches the type of trained model. For users interested in comparing our DAC framework with the **Metric3D** training framework, we have provided pre-trained weak baselines along with their testing scripts, as detailed in the last two rows of the table. -->


## Training

<!-- To train metric depth estimation models under the **DepthAnyCamera (DAC)** framework, you can run the default code for indoor training datasets as follows:

```bash
python scripts/train_dac.py --config-file configs/train/hm3d+taskonomy+hypersim/hm3d+taskonomy+hypersim_dac_r101.json --base-path datasets --distributed --model-name IDiscERP
```

If you wish to train with a larger backbone, use the following command:

```bash
python scripts/train_dac_large.py --config-file configs/train/hm3d+taskonomy+hypersim/hm3d+taskonomy+hypersim_dac_swinl_s2.json --base-path datasets --distributed --model-name IDiscERP
```

For users interested in comparing our DAC framework with the **Metric3D** training framework, the following command can be used:

```bash
python scripts/train_persp.py --config-file configs/train/hm3d+taskonomy+hypersim/hm3d+taskonomy+hypersim_r101.json --base-path datasets --distributed --model-name IDisc
```

The corresponding testing script can be found at [scripts/test_persp.py](scripts/test_persp.py).

Similar commands apply to outdoor model training. There are various options available depending on the dataset or architecture. Interested users can refer to the table below for basic usage or consult the provided [launch.json](.vscode/launch.json) for convenient use or debugging in VSCode. We also provide all the training configurations we’ve used in [configs/train](configs/train).


| Training Target | Training script | --config-file | --model-name |
|:-|:-|:-|:-:|
| dac-indoor-resnet101          | scripts/train_dac.py           | [relative path](configs/train/hm3d+taskonomy+hypersim/hm3d+taskonomy+hypersim_dac_r101.json)       | IDiscERP or IDisc or CNNDepth  |
| dac-indoor-swinl              | scripts/train_dac_large.py     | [relative path](configs/train/hm3d+taskonomy+hypersim/hm3d+taskonomy+hypersim_dac_swinl_s2.json)   | IDiscERP or IDisc or CNNDepth  |
| dac-outdoor-resnet101         | scripts/train_dac.py           | [relative path](configs/train/ddad+lyft/ddad+lyft_dac_r101.json)                                   | IDiscERP or IDisc or CNNDepth  |
| dac-outdoor-swinl             | scripts/train_dac_large.py     | [relative path](configs/train/ddad+lyft/ddad+lyft_dac_swinl_s2.json)                               | IDiscERP or IDisc or CNNDepth  |
| metric3d-indoor-resnet101     | scripts/train_persp.py         | [relative path](configs/train/hm3d+taskonomy+hypersim/hm3d+taskonomy+hypersim_r101.json)           | IDisc or CNNDepth  |
| metric3d-indoor-swinl         | scripts/train_persp.py         | [relative path](configs/train/hm3d+taskonomy+hypersim/hm3d+taskonomy+hypersim_swinl.json)          | IDisc or CNNDepth  |
| metric3d-outdoor-resnet101    | scripts/train_persp.py         | [relative path](configs/train/ddad+lyft/ddad+lyft_r101.json)                                       | IDisc or CNNDepth  |
| metric3d-outdoor-swinl        | scripts/train_persp.py         | [relative path](configs/train/ddad+lyft/ddad+lyft_swinl_s2.json)                                   | IDisc or CNNDepth  | -->


## Acknowledgements
We thank the authors of the following awesome codebases:
- [DAC](https://github.com/yuliangguo/depth_any_camera)
- [UniK3D](https://github.com/lpiccinelli-eth/unik3d)
- [iDisc](https://github.com/SysCV/idisc)
- [Metric3D](https://github.com/YvanYin/Metric3D)
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)
- [OmniFusion](https://github.com/yuliangguo/OmniFusion)

<!-- For developers interested in multi-view-stereo designs contributing to the cross-camera generalization problem, we refer them to the following insightful works
- [nvTorchCam](https://github.com/NVlabs/nvTorchCam)
- [FoVA-Depth](https://research.nvidia.com/labs/lpr/fova-depth/) -->

## License
This software is released under MIT license. You can view a license summary [here](LICENSE).


## Citation

<!-- If you find our work useful in your research please consider citing our publication:
```bibtex
@inproceedings{Guo2025DepthAnyCamera,
  title={Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera},
  author={Yuliang Guo and Sparsh Garg and S. Mahdi H. Miangoleh and Xinyu Huang and Liu Ren},
  booktitle={CVPR},
  year={2025}
}
``` -->
