# Monocular Occupancy Prediction for Scalable Indoor Scenes

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
$ git clone https://github.com/hongxiaoy/ISO.git
$ cd ISO/
$ git clone https://github.com/LiheYoung/Depth-Anything.git depth_anything
$ pip install -r requirements.txt
```

> :bulb:Note
> 
> Change L140 in ```depth_anything/metric_depth/zoedepth/models/base_models/dpt_dinov2/dpt.py``` to
> 
> ```self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder), pretrained=False)```
>
> And download Depth-Anything pre-trained [model](https://github.com/LiheYoung/Depth-Anything/tree/main/metric_depth#evaluation) checkpoints file to ```checkpoints/```.

4. Install tbb:

```
$ conda install -c bioconda tbb=2020.2
```

5. Finally, install ISO:

```
$ pip install -e ./
```

> :bulb:Note
> 
> If you move the ISO dir to another place, you should run
>
> ```pip cache purge```
>
> then run ```pip install -e ./``` again.

### Datasets

#### NYUv2

1. Download the [NYUv2 dataset](https://www.rocq.inria.fr/rits_files/computer-vision/monoscene/nyu.zip).

2. Create a folder to store NYUv2 preprocess data at `/path/to/NYU/preprocess/folder`.

3. Store paths in environment variables for faster access:

```
$ export NYU_PREPROCESS=/path/to/NYU/preprocess/folder
$ export NYU_ROOT=/path/to/NYU/depthbin 
```

> :bulb:Note
> 
> Recommend using
> 
> ```echo "export NYU_PREPROCESS=/path/to/NYU/preprocess/folder" >> ~/.bashrc```
> 
> format command for future convenience.

4. Preprocess the data to generate labels at a lower scale, which are used to compute the ground truth relation matrices:

```
$ cd ISO/
$ python iso/data/NYU/preprocess.py NYU_root=$NYU_ROOT NYU_preprocess_root=$NYU_PREPROCESS
```

### Pretrained Models

Download ISO pretrained models [on NYUv2](https://huggingface.co/hongxiaoy/ISO/tree/main), then put them in the folder `/path/to/ISO/trained_models`.

## Running ISO

### Training

#### NYUv2

1. Create folders to store training logs at **/path/to/NYU/logdir**.

2. Store in an environment variable:

```
$ export NYU_LOG=/path/to/NYU/logdir
```

3.  Train ISO using 2 GPUs with batch_size of 4 (2 item per GPU) on NYUv2:
```
$ cd ISO/
$ python iso/scripts/train_iso.py \
    dataset=NYU \
    NYU_root=$NYU_ROOT \
    NYU_preprocess_root=$NYU_PREPROCESS \
    logdir=$NYU_LOG \
    n_gpus=2 batch_size=4
```

### Evaluating

#### NYUv2

To evaluate ISO on NYUv2 test set, type:

```
$ cd ISO/
$ python iso/scripts/eval_iso.py \
    dataset=NYU \
    NYU_root=$NYU_ROOT\
    NYU_preprocess_root=$NYU_PREPROCESS \
    n_gpus=1 batch_size=1
```

### Inference

Please create folder **/path/to/monoscene/output** to store the ISO outputs and store in environment variable:

```
export ISO_OUTPUT=/path/to/iso/output
```

#### NYUv2

To generate the predictions on the NYUv2 test set, type:

```
$ cd ISO/
$ python iso/scripts/generate_output.py \
    +output_path=$ISO_OUTPUT \
    dataset=NYU \
    NYU_root=$NYU_ROOT \
    NYU_preprocess_root=$NYU_PREPROCESS \
    n_gpus=1 batch_size=1
```

### Visualization

#### NYUv2 
```
$ cd ISO/
$ python iso/scripts/visualization/NYU_vis_pred.py +file=/path/to/output/file.pkl
```
