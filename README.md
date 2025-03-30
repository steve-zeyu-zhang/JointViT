<div align="center"><h1> JointViT: Modeling Oxygen Saturation Levels with Joint Supervision on Long-Tailed OCTA <br>
  <sub><sup><a href="https://miua2024.github.io/">MIUA 2024 Oral</a></sup></sub> 
</h1>

[Zeyu Zhang](https://steve-zeyu-zhang.github.io), [Xuyin Qi](https://www.linkedin.com/in/xuyin-q-29672524a/), [Mingxi Chen](https://www.linkedin.com/in/mingxi-chen-4b57562a1/), [Guangxi Li](https://github.com/lgX1123), [Ryan Pham](https://www.flinders.edu.au/people/ryan.pham), [Ayub Qassim](https://www.flinders.edu.au/people/ayub.qassim), [Ella Berry](https://www.linkedin.com/in/ella-berry-a2a3aab4/), [Zhibin Liao](https://researchers.adelaide.edu.au/profile/zhibin.liao), [Owen Siggs](https://researchnow.flinders.edu.au/en/persons/owen-siggs-2), [Robert Mclaughlin](https://researchers.adelaide.edu.au/profile/robert.mclaughlin), [Jamie Craig](https://www.flinders.edu.au/people/jamie.craig), [Minh-Son To](https://www.flinders.edu.au/people/minhson.to)

[![Website](https://img.shields.io/badge/Website-Demo-fedcba?style=flat-square)](https://steve-zeyu-zhang.github.io/JointViT/) [![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--031--66955--2__11-fcb520?style=flat-square&logo=doi)](https://doi.org/10.1007/978-3-031-66955-2_11) [![arXiv](https://img.shields.io/badge/arXiv-2404.11525-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2404.11525) [![Papers With Code](https://img.shields.io/badge/Papers%20With%20Code-555555.svg?style=flat-square&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB2aWV3Qm94PSIwIDAgNTEyIDUxMiIgd2lkdGg9IjUxMiIgIGhlaWdodD0iNTEyIiA+PHBhdGggZD0iTTg4IDEyOGg0OHYyNTZIODh6bTE0NCAwaDQ4djI1NmgtNDh6bS03MiAxNmg0OHYyMjRoLTQ4em0xNDQgMGg0OHYyMjRoLTQ4em03Mi0xNmg0OHYyNTZoLTQ4eiIgc3Ryb2tlPSIjMjFDQkNFIiBmaWxsPSIjMjFDQkNFIj48L3BhdGg+PHBhdGggZD0iTTEwNCAxMDRWNTZIMTZ2NDAwaDg4di00OEg2NFYxMDR6bTMwNC00OHY0OGg0MHYzMDRoLTQwdjQ4aDg4VjU2eiIgc3Ryb2tlPSIjMjFDQkNFIiBmaWxsPSIjMjFDQkNFIj48L3BhdGg+PC9zdmc+)](https://paperswithcode.com/paper/jointvit-modeling-oxygen-saturation-levels) [![BibTeX](https://img.shields.io/badge/BibTeX-Citation-eeeeee?style=flat-square)](https://steve-zeyu-zhang.github.io/JointViT/static/scholar.html)

</div>

_The oxygen saturation level in the blood (SaO<sub>2</sub>) is crucial for health, particularly in relation to sleep-related breathing disorders. However, continuous monitoring of SaO<sub>2</sub> is time-consuming and highly variable depending on patients' conditions. Recently, optical coherence tomography angiography (OCTA) has shown promising development in rapidly and effectively screening eye-related lesions, offering the potential for diagnosing sleep-related disorders. To bridge this gap, our paper presents three key contributions. Firstly, we propose <b>JointViT</b>, a novel model based on the Vision Transformer architecture, incorporating a <b>joint loss</b> function for supervision. Secondly, we introduce a <b>balancing augmentation</b> technique during data preprocessing to improve the model's performance, particularly on the long-tail distribution within the OCTA dataset. Lastly, through comprehensive experiments on the OCTA dataset, our proposed method significantly outperforms other state-of-the-art methods, achieving improvements of up to <b>12.28%</b> in overall accuracy. This advancement lays the groundwork for the future utilization of OCTA in diagnosing sleep-related disorders._

![main](https://github.com/steve-zeyu-zhang/JointViT/blob/website/static/images/jointvit.svg)

## News

<b>(06/18/2024)</b> &#127881; Our paper has been selected as an <b style="color: red;">oral presentation</b> at <a href="https://miua2024.github.io/"><b>MIUA 2024</b></a>!


<b>(05/14/2024)</b> &#127881; Our paper has been accepted to <a href="https://miua2024.github.io/"><b>MIUA 2024</b></a>!

## Hardware
NVIDIA GeForce GTX TITAN X 

## Environment

For docker container:

```
docker pull qiyi007/oct:1.0
```
For dependencies:

```
conda create -n jointvit
```
```
conda activate jointvit
```
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=12.0 -c pytorch
```
## Dataset

File directories as follows(3fold)

```
|-- OCT-Code
|   |-- util
       |-- ...
|   |-- vit_pytorch
       |-- ...
|   |-- train.py
|   |-- train_0 <split fold index information file>
|   |-- train_1 <split fold index information file>
|   |-- train_2 <split fold index information file>
|   |-- test_0 <split fold index information file>
|   |-- test_1 <split fold index information file>
|   |-- test_2 <split fold index information file>
|   |-- Sleep-results.xlsx <label information file>
|   |-- images-with-labels <dataset folder>
|       |-- ...

```

OR:

```
|-- OCT-Code
|   |-- util
       |-- ...
|   |-- vit_pytorch
       |-- ...
|   |-- train.py
|   |-- train_data <dataset folder>
|       |-- ...
|   |-- test_data <dataset folder>
|       |-- ...

```
## Training
### modify args dict in train.py
```
 args = {
                    'device': torch.device("cuda:1"),
                    # 'model': get_model_octa_resume(outsize=5, path='ckpt_path', dropout=0.15),
                    # 'model': get_model_conv(pretrain_out=4,outsize=5, path='/OCT-Covid/covid_ckpts/oct4class_biglr/val_acc0.9759836196899414.pt'),
                    'model': get_vani(outsize=5, dropout=0.25),
                    # 'model': get_model_oct_withpretrain(pretrain_out=4,outsize=5, path='/OCT-Covid/covid_ckpts/oct4class_biglr/val_acc0.9759836196899414.pt', dropout=0.15),
                    'save_path': 'save_path', 
                    'bce_weight': 1,     
                    'epochs': 200, 
                    'lr': lr, 
                    'batch_size': 300, 
                    'datasets': get_dataUNI(split_idx=split, aug_class=isaug, bal_val = isbalval),
                    'vote_loader': DataLoader(get_dataUNI(split_idx=split, aug_class=isaug, bal_val = isbalval, infer_3d=True)[1], batch_size=1, shuffle=False),
                    'is_echo': False,
                    'optimizer': optim.Adam,
                    'scheduler': optim.lr_scheduler.CosineAnnealingLR,
                    'train_loader': None,
                    'eval_loader': None,
                    'shuffle': True,
                    'is_MIX': True, # use mixloss input
                    'wandb': ['wandb account','project name',run_name],
                    'decay': 1e-3,
                }
```
3-fold training & validation:
```
python train_3fold.py
```
Default training:

```
python train.py
```
After running train.py, the metrics for a test fold will be displayed.


## Citation

```
@inproceedings{zhang2024jointvit,
  title={Jointvit: Modeling oxygen saturation levels with joint supervision on long-tailed octa},
  author={Zhang, Zeyu and Qi, Xuyin and Chen, Mingxi and Li, Guangxi and Pham, Ryan and Qassim, Ayub and Berry, Ella and Liao, Zhibin and Siggs, Owen and Mclaughlin, Robert and others},
  booktitle={Annual Conference on Medical Image Understanding and Analysis},
  pages={158--172},
  year={2024},
  organization={Springer}
}
```
