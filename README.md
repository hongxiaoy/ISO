# Monocular Occupancy Prediction for Scalable Indoor Scenes

[TOC]

## Preparing ISO

### Installation

1. Create conda environment:

```
$ conda create -n iso python=3.9 -y
$ conda activate iso
```
2. This code was implemented with python 3.9, pytorch 2.0.0 and CUDA 11.7. Please install [PyTorch](https://pytorch.org/): 

```
$ conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install the additional dependencies:

```
$ cd ISO/
$ pip install -r requirements.txt
```

4. Install tbb:

```
$ conda install -c bioconda tbb=2020.2
```

5. Finally, install ISO:

```
$ pip install -e ./
```

### Datasets

#### NYUv2

1. Download the [NYUv2 dataset](https://www.rocq.inria.fr/rits_files/computer-vision/monoscene/nyu.zip).

2. Create a folder to store NYUv2 preprocess data at `/path/to/NYU/preprocess/folder`.

3. Store paths in environment variables for faster access:

```
$ export NYU_PREPROCESS=/path/to/NYU/preprocess/folder
$ export NYU_ROOT=/path/to/NYU/depthbin 
```

4. Preprocess the data to generate labels at a lower scale, which are used to compute the ground truth relation matrices:

```
$ cd ISO/
$ python iso/data/NYU/preprocess.py NYU_root=$NYU_ROOT NYU_preprocess_root=$NYU_PREPROCESS
```

### Pretrained Models

Download ISO pretrained models [on NYUv2](https://huggingface.co/hongxiaoy/ISO/tree/main), then put them in the folder `/path/to/ISO/trained_models`.