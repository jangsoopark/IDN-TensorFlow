# VDSR-TensorFlow

---



## Introduction

This repository is TensorFlow implementation of VDSR (CVPR16). 

You can see more details from paper and author's project page

- Project page : [VDSR page](<https://cv.snu.ac.kr/research/VDSR/>)

- Paper : ["Accurate Image Super-Resolution Using Very Deep Convolutional Network"](<https://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf>)

---

## Network Structure

>  VDSR-TensorFlow/model/network.py

![VDSR Network Structure](./resources/figure/001-VDSR.png)

- ILR denotes Interpolated Low Resolution image
- SR denotes reconstructed super resolution image



### Details

- VDSR structures

| Layer (# layers)  | Filter size | Input Dimension | Output Dimension | Activation Function |
| ----------------- | ----------- | --------------- | ---------------- | ------------------- |
| Input Layer (1)   | $3\times 3$ | 1               | 64               | ReLU                |
| Hidden Layer (18) | $3\times 3$ | 64              | 64               | ReLU                |
| Output Layer (1)  | $3\times 3$ | 64              | 1                | -                   |

- I/O 

| Input (LR)                                     | Output (Residual)                              | Reconstructed (LR + Residual)                  |
| ---------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| ![LR](./resources/figure/002-1-LR_scale_2.png) | ![HF](./resources/figure/002-2-HF_scale_2.png) | ![SR](./resources/figure/002-3-SR_scale_2.png) |

---

## Training

### Loss Function

> \_loss_function(self, reg_parameter) in VDSR-TensorFlow/model/\_\__init\_\_.py

- Basic loss function


$$
Loss(W)=\frac{1}{2}||y-f(x)||^{2}
$$


- Loss functions for **residual learning**

$$
Loss(W)=\frac{1}{2}||r-f(x)||^{2}
$$

- Regularization

  - L2 regularization

  $$
  reg(W)=\frac{\lambda}{2}\sum_{w \in W} {||w||^{2}}
  $$

  

- Notations
  - $W$ : Weights in VDSR
  - $y$ : ground truth (original high resolution image, HR)
  - $x$ : interpolated low resolution image (ILR)
  - $f(x)$ : reconstructed super resolution image
  - $r$ : residual between HR and ILR
    - $r = y-x$
  - $\lambda$ : regularization parameter
    -  $\lambda$ : 0.0001



### Optimization

> \_optimization_function(self, grad_clip, momentum_rate) in VDSR-TensorFlow/model/\_\__init\_\_.py

- Optimization Method 

  - Stochastic Gradient Descent (SGD) method [[Wikipedia]](<https://en.wikipedia.org/wiki/Stochastic_gradient_descent>)
    - Momentum : 0.9

- Weight Initialization

  - He initialization [[paper]](<https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf>)

- **Learning Rate**

  - Extremely high value is used to speed-up convergence rate
  - Initial Learning rate : 0.1

- **Learning Rate Decay**

   ![learning rage in training](./resources/figure/003-learning_rate.png)

  - Learning rate is decreased by factor of 10 for every 20 epochs

- **Adjustable Gradient Clipping**

  - Clip individual gradients to $[-\frac{\theta}{\gamma}, \frac{\theta}{\gamma}]$
    - $\theta$ denotes parameters for gradient clipping
    - $\gamma$ denotes learning rate

- Epochs : 80

---

## Data Set

### Training Data


> VDSR-TensorFlow/data/generate_dataset/train_data.m

- 291 images
  - Download from Author's page [[zip(train)]](https://cv.snu.ac.kr/research/VDSR/train_data.zip)
- Bicubic interpolation is used for LR data acquisition
- Data Augmentations (Rotation, flip) were used
- Scale Factor : $\times 2$, $\times 3$, $\times 4$
- Patch size : 41
- Batch size : 64



### Testing Data

>  VDSR-TensorFlow/data/generate_dataset/test_data.m

- Set5, Set14, B100, Urban100
  - Download from Author's page [[zip(test)]](https://cv.snu.ac.kr/research/VDSR/test_data.zip)
- Bicubic interpolation is used for LR data acquisition
- Scale Factor : $\times 2$, $\times 3$, $\times 4$

---

## Results

### Validation

PSNR performance plot on Set5

| ![scale2](./resources/figure/004-1-validation_Set5_scale_2.png) | ![scale2](./resources/figure/004-2-validation_Set5_scale_3.png) | ![scale2](./resources/figure/004-3-validation_Set5_scale_4.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Scale 2                                                      | Scale 3                                                      | Scale 4                                                      |



### Objective Quality Assessment

#### Methods

- Bicubic Interpolation 
  - imresize(..., ..., 'bicubic') in Matlab
- VDSR (Original)
  - Author's MatConvNet implementation [[Code]](https://cv.snu.ac.kr/research/VDSR/VDSR_code.zip)
- VDSR (TensorFlow)
  - TensorFlow implementation
  - Train Details for Comparison
    - Gradient Clipping parameter $\theta$ = 0.0001
    - Data Augmentation
      - Rotation : 90°
      - Flip : left / right


#### Average PSNR/SSIM

- **Set5**

| scale      | Bicubic        | VDSR (Original) | VDSR(TensorFlow) |
| ---------- | -------------- | --------------- | ---------------- |
| $\times 2$ | 33.68 / 0.9304 | 37.53 / 0.9586  | 37.07 / 0.9576   |
| $\times 3$ | 30.40 / 0.8682 | 33.66 / 0.9213  | 33.20 / 0.9171   |
| $\times 4$ | 28.43 / 0.8104 | 31.35 / 0.8838  | 30.90 / 0.8756   |



- **Set14**

| scale      | Bicubic        | VDSR (Original) | VDSR(TensorFlow) |
| ---------- | -------------- | --------------- | ---------------- |
| $\times 2$ | 30.24 / 0.8693 | 37.53 / 0.9586  | 32.67 / 0.9108   |
| $\times 3$ | 27.54 / 0.7746 | 33.66 / 0.9213  | 29.58 / 0.8295   |
| $\times 4$ | 26.00 / 0.7029 | 31.35 / 0.8838  | 27.81 / 0.7627   |

- **B100**

| scale      | Bicubic        | VDSR (Original) | VDSR(TensorFlow) |
| ---------- | -------------- | --------------- | ---------------- |
| $\times 2$ | 29.56 / 0.8442 | 37.53 / 0.9586  | 31.65 / 0.8943   |
| $\times 3$ | 27.21 / 0.7401 | 33.66 / 0.9213  | 28.66 / 0.7952   |
| $\times 4$ | 25.96 / 0.6697 | 31.35 / 0.8838  | 27.14 / 0.7217   |

- **Urban100**

| scale      | Bicubic        | VDSR (Original) | VDSR(TensorFlow) |
| ---------- | -------------- | --------------- | ---------------- |
| $\times 2$ | 26.88 / 0.8410 | 37.53 / 0.9586  | 30.20 / 0.9087   |
| $\times 3$ | 24.46 / 0.7358 | 33.66 / 0.9213  | 26.69 / 0.8178   |
| $\times 4$ | 23.14 / 0.6588 | 31.35 / 0.8838  | 24.85 / 0.7406   |

---

### Visual Quality

- "img002" of Urban100 for scale factor $\times 2$

| Ground Truth                                                 | Bicubic                                                      | VDSR(TensorFlow)                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![img002_gt](./resources/figure/005-1-visual_img002_gt.png)  | ![img002_gt](./resources/figure/005-2-visual_img002_lr.png)  | ![img002_gt](./resources/figure/005-3-visual_img002_sr.png)  |
| ![img002_gt](./resources/figure/005-1-visual_img002_gt_part.png) | ![img002_gt](./resources/figure/005-2-visual_img002_lr_part.png) | ![img002_gt](./resources/figure/005-3-visual_img002_sr_part.png) |



- ???
- ???
- ???





---

## Usage

> **On Windows**
>
> - run.bat
>
> **On Linux** 
>
> - run.sh



### Training Command

- in run.bat/sh
  - python main.py --model_name=vdsr **--is_train=True** --grad_clip=1e-3

    

If you want to change other parameters for training, please see the file

> VDSR-TensorFlow/model/configurations.py 



### Testing Command

in run.bat

python main.py --model_name=vdsr **--is_train=False**



### Trained checkpoint in experiments

- checkpoint [[download]](https://drive.google.com/file/d/1wiej51wFY0oYsoKF7gGiWZT5_t5mFt0f/view?usp=sharing)

### Training dataset

- vdsr_train.h5 [[download]](https://drive.google.com/file/d/1wiej51wFY0oYsoKF7gGiWZT5_t5mFt0f/view?usp=sharing)

