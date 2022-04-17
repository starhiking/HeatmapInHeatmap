# HeatmapInHeatmap
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hih-towards-more-accurate-face-alignment-via/face-alignment-on-wflw)](https://paperswithcode.com/sota/face-alignment-on-wflw?p=hih-towards-more-accurate-face-alignment-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hih-towards-more-accurate-face-alignment-via/face-alignment-on-cofw)](https://paperswithcode.com/sota/face-alignment-on-cofw?p=hih-towards-more-accurate-face-alignment-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hih-towards-more-accurate-face-alignment-via/face-alignment-on-300w)](https://paperswithcode.com/sota/face-alignment-on-300w?p=hih-towards-more-accurate-face-alignment-via)

For models not using extra training data, HIH has got Rank 1 on [WFLW Leaderboard](https://paperswithcode.com/sota/face-alignment-on-wflw), Rank 1 on [COFW Leaderboard](https://paperswithcode.com/sota/face-alignment-on-cofw), Rank 3 on [300W Leaderboard](https://paperswithcode.com/sota/face-alignment-on-300w).

Arxiv:[HIH:Towards More Accurate Face Alignment via Heatmap in Heatmap](https://arxiv.org/abs/2104.03100) 

ICCV Workshops:[Revisting Quantization Error in Face Alignment](https://openaccess.thecvf.com/content/ICCV2021W/MFR/papers/Lan_Revisting_Quantization_Error_in_Face_Alignment_ICCVW_2021_paper.pdf)


It is the Pytorch Implementation of HIH:Towards More Accurate Face Alignment via Heatmap in Heatmap.

## Update Logs

### April 18, 2022

* We released HIH v2 in arxiv.

### April 17, 2022

* Pretrained Model and evaluation code on WFLW dataset are released.

### March 22, 2022

* HIH breaks the new records on WFLW and COFW.

### August 13, 2021

* ICCV Workshops Accept.

### April 03, 2021

* We released HIH v1 in arxiv.

## Introduction


## Results


## Installation

Download the dataset and checkpoint at [Baidu os](https://pan.baidu.com/s/1QESuPhP9d6TUVySNBqQcZw) (passwd:cjap)


The folder structure is as following:




## Run Evaluation on WFLW

Evaluation cmd:

    python tools/test_all.py --config_file experiments/Data_WFLW/HIHC_64x8_hg_l2.py --resume_checkpoint Best/WFLW/WFLW.pth

## Future Plans
- [x] Release evaluation code and pretrained model on WFLW dataset.

- [ ] Release training code on WFLW dataset.
 
- [ ] Release pretrained model and code on 300W and COFW dataset.

- [ ] Release facial landmark detection API

## Citations
If you find this useful for your research, please cite the following papers.


## Acknowledgments
This repository borrows or partially modifies hourglass model and data processing code from [Hourglass](https://github.com/raymon-tian/hourglass-facekeypoints-detection) [HRNet](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)

## License
This repository is released under the Apache 2.0 license.



