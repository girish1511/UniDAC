
<div align="center">
<h1>UniDAC: Universal Metric Depth Estimation for Any Camera</h1>

[**Girish Chandar Ganesan**](https://girish1511.github.io/)<sup>1</sup> . [**Yuliang Guo**](https://yuliangguo.github.io/)<sup>2</sup> · [**Liu Ren**](https://www.liu-ren.com/)<sup>2</sup> . [**Xiaoming Liu**](https://cs.unc.edu/person/xiaoming-liu/)<sup>1,3</sup>

<sup>1</sup>Michigan State University&emsp;&emsp;&emsp;<sup>2</sup>Bosch Research North America&emsp;&emsp;&emsp;<sup>3</sup>University of North Carolina at Chapel Hill


<a href='https://arxiv.org/abs/2603.27105'><img src='https://img.shields.io/badge/arXiv-UniDAC-red' alt='Paper PDF'></a>
<a href='https://girish1511.github.io/UniDAC/'><img src='https://img.shields.io/badge/Project_Page-UniDAC-green' alt='Project Page'></a>
<a href='https://huggingface.co/girish1511/UniDAC'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow'></a>

[**CVPR 2026**](https://cvpr.thecvf.com/Conferences/2026)

</div>

<p align="center">
  <img src="docs/pano_teaser.gif" alt="animated" />
</p>


## News


- [ ] Training code for UniDAC.
- [ ] Demo code for images with unknown camera parameters.
- [x] Demo code for easy setup and usage.
- [x] `2026-03-13`: Release of UniDAC checkpoint trained on moderately sized datasets.
- [x] `2026-03-13`: Testing and evaluation pipeline for zero-shot metric depth estimation on perspective, fisheye, and 360-degree datasets.
- [x] `2026-03-13`: Data preparation and curation scripts.
- [x] `2026-02-20`: UniDAC accepted by CVPR 2026!

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

## Demo

We provide a simple ready-to-run demo script in the `demo` folder along with the required sample inputs in `demo/input`.
`demo/demo_unidac.py` demonstrates the inference pipeline for diverse camera types and scenes, including ScanNet++(Indoor, Fisheye), Matterport3D(Indoor, 360) and KITTI360(Outdoor, Fisheye), using a unified model trained only on perspective images.

Download the checkpoint from <a href='https://huggingface.co/girish1511/UniDAC'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow'></a> and place in `checkpoints/`.
You can then run the demo script by running the following command and the visualizations will be stored in `demo/output`:
```
bash demo.sh
```

## Testing

Download the checkpoint from <a href='https://huggingface.co/girish1511/UniDAC'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow'></a> and place in `checkpoints/`.

Run the following to evaluate and reproduce the results presented in the paper:

```bash
bash eval.sh <domain> <dataset>
```

Different config files for evaluating the reported testing datasets are included in [configs/test](configs/test). Refer to the table below to set the `<domain>` and `<dataset>` arguments, which together select the corresponding configuration file for the dataset you wish to evaluate.

| |  ScanNet++  | Matterport3D | Pano3D-GibsonV2 |  KITTI-360 |   KITTI   |    NYU   |  NuScenes  |  iBims-1 |
| :---------: | :---------: | :----------: | :-------------: | :--------: | :-------: | :------: | :--------: | :------: |
|  `<domain>` |   `indoor`  |   `indoor`   |     `indoor`    |  `outdoor` | `outdoor` | `indoor` |  `outdoor` | `indoor` |
| `<dataset>` | `scannetpp` |     `gv2`    |   `scannetpp`   | `kitti360` |  `kitti`  |   `nyu`  | `nuscenes` |  `ibims` |



## Acknowledgements
We thank the authors of the following awesome codebases:
- [DAC](https://github.com/yuliangguo/depth_any_camera)
- [UniK3D](https://github.com/lpiccinelli-eth/unik3d)
- [iDisc](https://github.com/SysCV/idisc)
- [Metric3D](https://github.com/YvanYin/Metric3D)
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)
- [OmniFusion](https://github.com/yuliangguo/OmniFusion)

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
