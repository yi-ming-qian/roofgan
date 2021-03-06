# Roof-GAN: Learning to Generate Roof Geometry and Relations for Residential Houses

Code for the [CVPR 2021 paper](https://arxiv.org/abs/2012.09340) by [Yiming Qian](https://yi-ming-qian.github.io/), [Hao Zhang](https://www.cs.sfu.ca/~haoz/), and [Yasutaka Furukawa](https://www.cs.sfu.ca/~furukawa/). Supplementary document is [here](https://drive.google.com/file/d/130p1PjD2OuV6bhYdp8c_QEU6LPbUux8Q/view?usp=sharing).

## Getting Started

Clone the repository:
```bash
git clone https://github.com/yi-ming-qian/roofgan.git
```

We use Python 3.7 and PyTorch 1.2.0 in our implementation, please install dependencies:
```bash
conda create -n roofgan python=3.7
conda activate roofgan
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install -c conda-forge opencv
pip install -r requirements.txt
```

## Dataset
Please download our dataset from [here](https://www.dropbox.com/s/a6nc146y86uvyy9/normalmap.zip?dl=0). Then, change the option "data_root" in both "scripts/train_gan.sh" and "scripts/test_gan.sh" to the directory containing the dataset.

## Training
Run the following command to train Roof-GAN:
```bash
sh ./scripts/train_gan.sh
```

## Generation
Run the following command to generate roof models:
```bash
sh ./scripts/test_gan.sh
```

The generated models will be saved under "experiments/{proj_dir}/results/". We also provide our pre-trained models [here](https://www.dropbox.com/s/qxt0ek0kfcaq3pi/ckpt_epoch200000.pth?dl=0). To use it, please place it under "experiments/{proj_dir}/model_gan/".

## Evaluation
Run the following commands to evaluate with the RMMD and FID metrics:
```bash
cd evaluate
sh evaluate_RMMD.sh
cd fid
sh evaluate_fid.sh
```

Our generated results can be downloaded from [here](https://www.dropbox.com/s/urkdlznm876drjc/results.zip?dl=0), which should be placed at "experiments/" after unzipping.

## Contact
[https://yi-ming-qian.github.io/](https://yi-ming-qian.github.io/)

## Acknowledgements
We thank the authors of [PQ-Net](https://github.com/ChrisWu1997/PQ-NET) and of [House-GAN](https://github.com/ennauata/housegan). Parts of our implementation are modified based on their codes. The FID metric implmentation is copied from [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid).