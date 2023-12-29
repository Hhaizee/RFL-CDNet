# **RFL-CDNet: Towards Accurate** Change Detection via Richer Feature Learning

This software implements RFL-CDNet: Towards Accurate Change Detection via Richer Feature Learning in PyTorch. For more details, please refer to our paper 

## Abstract

​    Change Detection is a crucial but extremely challenging task of remote sensing image analysis, and much progress has been made with the rapid development of deep learning. However, most existing deep learning-based change detection methods mainly focus on intricate feature extraction and multi-scale feature fusion, while ignoring the insufficient utilization of features in the intermediate stages, thus resulting in sub-optimal results. To this end, we propose a novel framework, named RFL-CDNet, to utilize richer feature learning for change detection. Specifically, we improve the capability and utilization of feature learning via introducing deep supervision information at the intermediate stages. Furthermore, we design the Coarse-To-Fine Guiding (C2FG) module and the Learnable Fusion (LF) module to further improve feature learning and learn more discriminative feature representations. The C2FG module aims to seamlessly integrate the side output from previous coarse-scale into the current fine-scale prediction in a coarse-to-fine manner, while LF module assumes that the contribution of each stage and each spatial location is independent, thus designing a learnable module to fuse multiple predictions. Experiments on several benchmark datasets show that our proposed RFL-CDNet achieves state-of-the-art performance.

![image-20230828104853096](./image/architecture_of_model.png)

## Installation

Install [PyTorch](http://pytorch.org/) 1.7.1+ and other dependencies:

```
pip/conda install pytorch>=1.7.1, tqdm, tensorboardX, opencv-python, pillow, numpy, sklearn
```

## Run demo

Generate the train.txt, val.txt and test.txt

```
python write_path.py
```

A demo program can be found in demo. Before running the demo, download our pretrained models and best models from [Baidu Netdisk](https://pan.baidu.com/s/1k_FPHtNttV2mBsJ-M0ukRw?pwd=emby ) (Extraction code: emby) . Then launch demo by:

```
python eval.py
```

## Evaluatioin

```
python eval.py
```

```
python visualization.py
```

## Train a new model

Generate the train.txt, val.txt and test.txt:

```
python write_path.py
```

Submit the train.sh:

```
sbatch train.sh
```

## Results

>  Here gives some examples of change detection results, comparing with existing methods on CDD Dataset in Figure (a), and Figure(b) is the results on WHU Dataset.  

|           (a)           |           (b)           |
| :---------------------: | :---------------------: |
| ![CDD](./image/CDD.png) | ![WHU](./image/WHU.png) |

Evaluation of RFL-CDNet on different datasets with SNUNet, STANet, and DASNet as baseline:

<img src=".\image\table_3.jpg" style="zoom: 67%;" />

​                                                            **Tabel 1. WHU Cultivated Land Dataset** 

<img src=".\image\table_1.jpg" style="zoom: 67%;" />

​                                                                             **Tabel 2. WHU Dataset** 

<img src=".\image\table_2.jpg" style="zoom: 67%;" />

​														      				**Tabel 3. CDD Dataset** 

## Acknowledgements

The authors would like to thank the developers of PyTorch, SNUNet, STANet, and DASNet. 
Please let me know if you encounter any issues.

