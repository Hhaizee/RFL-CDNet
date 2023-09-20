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

Download the datasets from [Baidu Netdisk](https://pan.baidu.com/s/1vhF0r93TxPsOgxxJwcmUDA)(Extraction code:rvbv).

Generate the train.txt, val.txt and test.txt

```
python write_path.py
```

A demo program can be found in demo. Before running the demo, download our pretrained models from [Baidu Netdisk](https://pan.baidu.com/s/1y4GRIUWXh8eNvsy93Z2Smg) (Extraction code: eu68) or [Google drive](https://drive.google.com/drive/folders/13bp7FbOBUtQi_zkhE6GpetRgjms9TuF9?usp=sharing). Set the path of files  in tmp/***.pt. Then launch demo by:

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

Evaluation of SDACD on different datasets with SNUNet, STANet, and DASNet as baseline:

| **Methods**  |                          |           CDD            |                          |                           |       WHU building       |                          |
| :----------: | :----------------------: | :----------------------: | :----------------------: | :-----------------------: | :----------------------: | :----------------------: |
|              |         **P(%)**         |         **R(%)**         |         **F(%)**         |         **P(%)**          |         **R(%)**         |         **F(%)**         |
|    FC-EF     |          84.68           |          65.13           |          73.63           |           80.75           |          67.29           |          73.40           |
| FC-Siam-diff |          87.57           |          66.69           |          75.07           |           48.84           |          88.96           |          63.06           |
| FC-Siam-conc |          88.81           |          62.20           |          73.16           |           54.20           |          81.34           |          65.05           |
|    STANet    |          83.17           |          92.76           |          87.70           |           82.12           |          89.19           |          83.40           |
| SDACD-STANet |  87.40   **↑****4.23**   |   89.50  **↓****3.26**   |   88.40  **↑****0.70**   |   90.90  **↑****8.78**    | **93.50**  **↑****4.31** |   92.21  **↑****8.81**   |
|    DASNet    |          93.28           |          89.91           |          91.57           |           83.77           |          91.02           |          87.24           |
| SDACD-DASNet |   92.85  **↓****0.43**   |   91.87  **↑****1.96**   |   92.35  **↑****0.78**   |   89.21  **↑****5.44**    |   90.46  **↓****0.56**   |   89.83  **↑****2.59**   |
|    SNUNet    |          96.60           |          94.77           |          95.68           |           82.12           |          89.19           |          85.51           |
| SDACD-SNUNet | **97.13**  **↑****0.53** | **97.56**  **↑****2.79** | **97.34**  **↑****1.66** | **93.85**  **↑****11.73** |   90.91  **↑****1.72**   | **92.36**  **↑****6.85** |

The grid search results of λf and λCD. Here we fixed λcyc=10 and λi=1.

| Baseline |  λf  | λCD  | P(%)  | R(%)  |   F(%)    |
| :------: | :--: | :--: | :---: | :---: | :-------: |
|          |  1   | 0.05 | 93.38 | 90.85 |   92.10   |
|          |  1   | 0.1  | 93.85 | 90.91 | **92.36** |
|  SNUNet  |  1   | 0.2  | 93.92 | 90.94 |   92.09   |
|          | 0.5  | 0.1  | 93.31 | 91.28 |   92.28   |
|          |  1   | 0.1  | 93.85 | 90.91 | **92.36** |
|          |  2   | 0.1  | 94.56 | 89.99 |   92.22   |

## Acknowledgements

The authors would like to thank the developers of PyTorch, SNUNet, STANet, and DASNet. 
Please let me know if you encounter any issues.

