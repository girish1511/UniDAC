## Dataset Preparation Guidelines

<!-- ![data](table_data_coverage.png) -->
Our training set builds on top of the training set of [Depth Any Camera](https://github.com/yuliangguo/depth_any_camera/tree/main).
Follow DAC's [data preparation](https://github.com/yuliangguo/depth_any_camera/blob/main/docs/DATA.md) guide to prepare **HM3D**, **HyperSim**, **Taskonomy**, **DDAD**, **Lyft**, **KITTI** and **NYU** datasets and ensure they are in the required data structure format.

We provide below the data preparation steps, i.e., **Argoverse2**, **A2D2**, **iBims** and **NuScenes**. The required data format for these datasets are as follows:

```bash
depth_any_camera
├── datasets
│      ├── hm3d
│      ├── hypersim
│      ├── taskonomy
│      ├── scannetpp
│      ├── gibson_v2
│      ├── matterport3d
│      ├── ddad
│      ├── lyft
│      ├── kitti360
│      ├── kitti
│      ├── nyu
│      ├── a2d2
│      │     ├── camera_lidar
│      │     │     ├── 20180810_150607
│      │     │     │     ├── camera
│      │     │     │     ├── depth
│      │     │     ├── 20190401_121727
│      │     │     ├── ...
│      ├── argoverse2
│      │     ├── train
│      │     │     ├── 5d391e54-adec-3584-adf0-5025d7564e1b
│      │     │     │     ├── calibration
│      │     │     │     ├── sensors
│      │     │     │     │     ├──cameras
│      │     │     │     │     │     ├──ring_*
│      │     │     │     │     ├──lidar
│      │     │     ├── ...
│      │     ├── depth
│      │     │     ├── 5d391e54-adec-3584-adf0-5025d7564e1b
│      │     │     │     ├──ring_*
│      │     │     ├── ...
│      │     ├── ...
│      ├── ibims
│      │     ├── m1455541
│      │     │     ├── ibims1_core_raw
│      │     │     │     ├── rgb
│      │     │     │     ├── depth
│      │     │     │     ├── ...
│      ├── nuscenes
│      │     ├── samples
│      │     ├── depth
│      │     ├── point_info
│      │     ├── rgb
│      │     ├── ...
```

Ensure that the datasets are soft-linked to folders under the `datasets` directory and follow the data structure mentioned [here](https://github.com/yuliangguo/depth_any_camera/blob/main/docs/DATA.md), so that our config files and scripts can be directly utilized.

We provide below the data preparation steps and the file structure for the remaining datasets, i.e., **Argoverse2**, **A2D2**, **iBims** and **NuScenes**.

## Testing Datasets


### **iBims-1**
Download the official dataset from [here](https://www.asg.ed.tum.de/lmf/ibims1/) which includes RGB-D data samples.
The dataset is soft-linked in `datasets/ibims` and splits are saved [here](../splits/kitti)

### **NuScenes**
Download the official dataset from [here](https://www.nuscenes.org/nuscenes), which includes camera, lidar and calibration data for 1000 scenes.
Follow the instructions here to generate the depthmaps.
The dataset is soft-linked to `datasets/nuscenes`, and splits are saved [here](../splits/nuscenes).

---

## Training Datasets

The three major indoor datasets used for training are provided by [OmniData](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/dataset#readme). The package can be installed as follows:

```bash
conda install -c conda-forge aria2
pip install 'omnidata-tools'
```

### **Habitat-Matterport**
The Habitat-Matterport dataset can be downloaded by running:

```bash
omnitools.download point_info rgb depth_euclidean mask_valid --components hm3d --subset tiny --dest ./omnidata_hm3d --name your-name --email your-email --agree_all
```

We use the first 50 scenes to compose part of our indoor training data. Unfortunately, the full dataset (~8TB) must be downloaded before deleting the unused parts. After downloading, use the [script](splits/hm3d/prepare_hm3d_split_files_in_nyu_format.py) to generate split files.

The dataset is soft-linked to `datasets/hm3d`, and splits are saved [here](../splits/hm3d).

### **Taskonomy**
Omnidata provides the tiny version of Taskonomy, which we use for part of our indoor training data. Download it with:

```bash
omnitools.download point_info rgb depth_euclidean mask_valid --components taskonomy --subset tiny --dest ./omnidata_taskonomy_tiny --name your-name --email your-email --agree_all
```

After downloading, use the [script](../splits/taskonomy/prepare_taskonomy_split_files_in_nyu_format.py) to generate split files.

The dataset is soft-linked to `datasets/taskonomy`, and splits are saved [here](../splits/taskonomy).

### **Hypersim**
The full Hypersim dataset is used to compose part of our indoor training data. Download it with:

```bash
omnitools.download point_info rgb depth_euclidean mask_valid --components hypersim --subset fullplus --dest ./omnidata_hypersim --name your-name --email your-email --agree_all
```

After downloading, use the [script](../splits/hypersim/prepare_hypersim_split_files_in_nyu_format.py) to generate split files.

The dataset is soft-linked to `datasets/hypersim`, and splits are saved [here](../splits/hypersim).

### **DDAD**
Clone the [DDAD repository](https://github.com/TRI-ML/DDAD) and export it to your `PYTHONPATH`:

```bash
cd ..
git clone https://github.com/TRI-ML/dgp.git
export PYTHONPATH="$PWD/dgp:$PYTHONPATH"
```

Then run the code in `../splits/ddad` to download and process the dataset. Use the splits and info files from `../splits/ddad`. To download and process splits (e.g., train or val), use the following command:

```bash
cd ./depth_any_camera
python ./splits/ddad/get_ddad.py --base-path <BASE-PATH> --split <split-chosen>
```

The dataset is soft-linked to `datasets/ddad`, and splits and intrinsic parameters are saved [here](../splits/ddad).

### **LYFT**
The dataset can be downloaded from this [webpage](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data). After downloading, use the [script](../splits/lyft/generate_json_train.py) to prepare depth maps from the original object detection dataset for training, and another [script](../splits/lyft/generate_json_val.py) for validation. Use the [script](../splits/lyft/generate_splits.py) to generate NYU-style split files.

The dataset is soft-linked to `datasets/lyft`, and splits and intrinsic parameters are saved [here](../splits/lyft).

---