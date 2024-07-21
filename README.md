# Self-Supervised Contrastive Learning to Predict the Progression of Alzheimer's Disease with 3D Amyloid-PET
This repository provides the code to implement the following [paper](https://www.mdpi.com/2306-5354/10/10/1141):

Kwak, M. G., Su, Y., Chen, K., Weidman, D., Wu, T., Lure, F., ... & Alzheimer’s Disease Neuroimaging Initiative. (2023).
Self-Supervised Contrastive Learning to Predict the Progression of Alzheimer’s Disease with 3D Amyloid-PET.
<i>Bioengineering, 10</i>(10), 1141.

The proposed SMoCo, built upon Momentum Contrast (MoCo), is contrastive learning method to accurately predict the conversion to AD for individuals with mild cognitive impairment (MCI) with 3D amyloid-PET. 
SMoCo uses both labeled and unlabeled data to capture general semantic representations underlying the images.

## Requirements
### Installation
To use this package safely, ensure you have the following:
* Python 3.10+ environment
* PyTorch 2.0.0+

Additionally, we recommend using [wandb](https://wandb.ai/site) to track model training and evaluation. It can be enabled by `enable_wandb` argument.

## File Description
```
    ├── configs/                  # task-specific arguments
    ├── datasets/                 # pytorch Dataset and transformation functions
    ├── layers/                   # functions for SMoCo                     
    ├── models/                   # backbone and head: subnetworks of DenseNet and ResNet
    ├── tasks/                    # SMoCo, fine-tuning (w/ and w/o demographic information), and external evaluation on AIBL dataset                         
    ├── utils/                    # various functions for GPU setting, evaluation, optimization, and so on.
    ├── run_supmoco.py            # the main run code for SMoCo pre-training
    ├── run_finetune.py           # the main run code for SMoCo fine-tuning
    ├── run_demo_finetune.py      # the main run code for SMoCo fine-tuning with demographic information
    ├── run_classification.py     # the main run code for simple classification model
    └── run_aibl.py               # The main run code for blind evaluation on AIBL dataset
```

## Usage
### main run codes
1. Pre-train SMoCo with `run_supmoco.py`.
2. Fine-tune the pretrained SMoCo with `run_finetune.py` or `run_demo_finetune.py`. Make sure to correctly input the pre-trained checkpoint directory.
3. Other run codes are implemented for additional evaluations.

```
python run_supmoco.py
python run_finetune.py --pretrained_dir your_pretrained_dir
```

### Citation
If you use this project in your research, please cite it as follows:
```
@article{kwak2023self,
  title={Self-Supervised Contrastive Learning to Predict the Progression of Alzheimer’s Disease with 3D Amyloid-PET},
  author={Kwak, Min Gu and Su, Yi and Chen, Kewei and Weidman, David and Wu, Teresa and Lure, Fleming and Li, Jing and Alzheimer’s Disease Neuroimaging Initiative},
  journal={Bioengineering},
  volume={10},
  number={10},
  pages={1141},
  year={2023},
  publisher={MDPI}
}
```
