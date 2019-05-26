# IDN-TensorFlow

------

## Introduction

This repository is TensorFlow implementation of IDN(CVPR16). 

You can see more details from paper and author's project repository

- Github : [[IDN-Caffe]](https://github.com/Zheng222/IDN-Caffe)
- Paper : ["Fast and Accurate Single Image Super-Resolution via Information Distillation Network"](<http://openaccess.thecvf.com/content_cvpr_2018/papers/Hui_Fast_and_Accurate_CVPR_2018_paper.pdf>)

------

## Network Structure

> IDN-TensorFlow/model/network.py

![IDN Network Structure](./resources/figure/001-idn.png)

- **FBlock : Feqture extraction block**

  

![FBlock](./resources/figure/002-fblock.png)

- **DBlock : Information Distillation block**

  - Consists of **Enhancement unit** and **Compression unit**

    

![DBlock](./resources/figure/003-dblock.png)

- **RBlock : Reconstruction block**

  - De-convolution

    

![RBlock](./resources/figure/004-rblock.png)



- LR denotes Low Resolution image
- SR denotes reconstructed super resolution image

------

## Training

### Loss Function

> \_loss_function(self, reg_parameter) in IDN-TensorFlow/model/\_\__init\_\_.py

- Pre-training stage : L2 loss

<img src="https://tex.s2cms.ru/svg/%0ALoss(W)%3D%5Cfrac%7B1%7D%7B2%7D%7C%7Cy-f(x)%7C%7C%5E%7B2%7D%0A" alt="
Loss(W)=\frac{1}{2}||y-f(x)||^{2}
" />

- Fine tuning stage : L1 loss

<img src="https://tex.s2cms.ru/svg/%0ALoss(W)%3D%7Cr-f(x)%7C%0A" alt="
Loss(W)=|r-f(x)|
" />

- Regularization

  - L2 regularization

  <img src="https://tex.s2cms.ru/svg/%0A%20%20reg(W)%3D%5Cfrac%7B%5Clambda%7D%7B2%7D%5Csum_%7Bw%20%5Cin%20W%7D%20%7B%7C%7Cw%7C%7C%5E%7B2%7D%7D%0A%20%20" alt="
  reg(W)=\frac{\lambda}{2}\sum_{w \in W} {||w||^{2}}
  " />

  

- Notations

  - <img src="https://tex.s2cms.ru/svg/W" alt="W" /> : Weights in IDN
  - <img src="https://tex.s2cms.ru/svg/y" alt="y" /> : ground truth (original high resolution image, HR)
  - <img src="https://tex.s2cms.ru/svg/x" alt="x" /> : interpolated low resolution image (ILR)
  - <img src="https://tex.s2cms.ru/svg/f(x)" alt="f(x)" /> : reconstructed super resolution image
  - <img src="https://tex.s2cms.ru/svg/r" alt="r" /> : residual between HR and ILR
    - <img src="https://tex.s2cms.ru/svg/r%20%3D%20y-x" alt="r = y-x" />
  - <img src="https://tex.s2cms.ru/svg/%5Clambda%24%20%3A%20regularization%20parameter%0A%20%20%20%20-%20" alt="\lambda$ : regularization parameter
    - " />\lambda<img src="https://tex.s2cms.ru/svg/%20%3A%200.0001%0A%0A%0A%0A%23%23%23%20Optimization%0A%0A%3E%20%5C_optimization_function(self%2C%20grad_clip%2C%20momentum_rate)%20in%20IDN-TensorFlow%2Fmodel%2F%5C_%5C__init%5C_%5C_.py%0A%0A-%20Optimization%20Method%20%0A%0A%20%20-%20ADAM%20method%20%5B%5Bpaper%5D%5D(%3Chttps%3A%2F%2Farxiv.org%2Fpdf%2F1412.6980.pdf%3E)%0A%0A-%20Weight%20Initialization%0A%0A%20%20-%20He%20initialization%20%5B%5Bpaper%5D%5D(%3Chttps%3A%2F%2Fwww.cv-foundation.org%2Fopenaccess%2Fcontent_iccv_2015%2Fpapers%2FHe_Delving_Deep_into_ICCV_2015_paper.pdf%3E)%0A%0A-%20**Learning%20Rate**%0A%0A%20%20-%20Initial%20Learning%20rate%20%3A%201e-4%0A%0A-%20**Learning%20Rate%20Decay**%0A%0A%20%20-%20Learning%20rate%20decay%20is%20applied%20in%20tine%20tuning%20stage%0A%0A%20%20!%5Blearning%20rage%20in%20training%5D(.%2Fresources%2Ffigure%2F005-learning_rate.png)%0A%0A%20%20-%20Learning%20rate%20is%20decreased%20by%20factor%20of%2010%20for%20every%20250%20epochs%0A%0A-%20Epochs%20%0A%0A%20%20-%20Pre-training%20stage%3A%20100%0A%20%20-%20Fine%20tuning%20stage%20%3A%20600%0A%0A------%0A%0A%23%23%20Data%20Set%0A%0A%23%23%23%20Training%20Data%0A%0A%3E%20IDN-TensorFlow%2Fdata%2Fgenerate_dataset%2Ftrain_data.m%0A%0A-%20291%20images%0A%20%20-%20Download%20from%20Author's%20Repository%0A-%20Data%20Augmentations%20(Rotation%2C%20flip)%20were%20used%0A-%20Scale%20Factor%20%3A%20%24%5Ctimes%202%24%2C%20%24%5Ctimes%203%24%2C%20%24%5Ctimes%204%24%0A-%20Patch%20size%20%0A%0A%7C%20scale%20%7C%20Pre-Training%20(LR%20%2F%20GT)%20%7C%20Fine%20Tuning%20(LR%20%2F%20GT)%20%7C%0A%7C%20-----%20%7C%20----------------------%20%7C%20---------------------%20%7C%0A%7C%202%20%20%20%20%20%7C%2029%20%2F%2058%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%2039%20%2F%2078%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%203%20%20%20%20%20%7C%2015%20%2F%2045%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%2026%20%2F%2078%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%204%20%20%20%20%20%7C%2011%20%2F%2044%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%2019%20%2F%2076%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%0A-%20Batch%20size%20%3A%2064%0A%0A%0A%0A%23%23%23%20Testing%20Data%0A%0A%3E%20IDN-TensorFlow%2Fdata%2Fgenerate_dataset%2Ftest_data.m%0A%0A-%20Set5%2C%20Set14%2C%20B100%2C%20Urban100%0A%20%20-%20Download%20from%20Author's%20page%20%5B%5Bzip(test)%5D%5D(https%3A%2F%2Fcv.snu.ac.kr%2Fresearch%2FVDSR%2Ftest_data.zip)%0A-%20Bicubic%20interpolation%20is%20used%20for%20LR%20data%20acquisition%0A-%20Scale%20Factor%20%3A%20%24%5Ctimes%202%24%2C%20%24%5Ctimes%203%24%2C%20%24%5Ctimes%204%24%0A%0A------%0A%0A%23%23%20Results%0A%0A%23%23%23%20Validation%0A%0APSNR%20performance%20plot%20on%20Set5%0A%0A-%20Scale%202%0A%0A%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%20%20%7C%20------------%20%7C%20-----------%20%7C%0A%20%20%7C%20Pre-training%20%7C%20Fine%20Tuning%20%7C%0A%0A%20%20%0A%0A-%20Scale%203%0A%0A%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%20%20%7C%20------------%20%7C%20-----------%20%7C%0A%20%20%7C%20Pre-training%20%7C%20Fine%20Tuning%20%7C%0A%0A%20%20%0A%0A-%20Scale%204%0A%0A%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%20%20%7C%20------------%20%7C%20-----------%20%7C%0A%20%20%7C%20Pre-training%20%7C%20Fine%20Tuning%20%7C%0A%0A%20%20%0A%0A%23%23%23%20Objective%20Quality%20Assessment%0A%0A%23%23%23%23%20Methods%0A%0A-%20Bicubic%20Interpolation%20%0A%20%20-%20imresize(...%2C%20...%2C%20'bicubic')%20in%20Matlab%0A-%20IDN(Original)%0A%20%20-%20Author's%20Caffe%20implementation%20%5B%5BCode%5D%5D(https%3A%2F%2Fgithub.com%2FZheng222%2FIDN-Caffe)%0A-%20IDN%20(TensorFlow)%0A%20%20-%20TensorFlow%20implementation%0A%20%20-%20Train%20Details%20for%20Comparison%0A%20%20%20%20-%20Data%20Augmentation%0A%20%20%20%20%20%20-%20Rotation%20%3A%2090%C2%B0%2C%20180%C2%B0%2C%20270%C2%B0%0A%20%20%20%20%20%20-%20Flip%20%3A%20left%20%2F%20right%0A%20%20%20%20%20%20-%20Down%20scale%20%3A%200.9%2C%200.8%2C%200.7%2C%200.6%0A%0A%23%23%23%23%20Average%20PSNR%2FSSIM%0A%0A-%20**Set5**%0A-%20Pre-Training%0A%0A%7C%20scale%20%20%20%20%20%20%7C%20Bicubic%20%20%20%20%20%20%20%20%7C%20IDN%20(Original)%20%7C%20IDN%20(TensorFlow)%20%7C%0A%7C%20----------%20%7C%20--------------%20%7C%20--------------%20%7C%20----------------%20%7C%0A%7C%20" alt=" : 0.0001



### Optimization

&gt; \_optimization_function(self, grad_clip, momentum_rate) in IDN-TensorFlow/model/\_\__init\_\_.py

- Optimization Method 

  - ADAM method [[paper]](&lt;https://arxiv.org/pdf/1412.6980.pdf&gt;)

- Weight Initialization

  - He initialization [[paper]](&lt;https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf&gt;)

- **Learning Rate**

  - Initial Learning rate : 1e-4

- **Learning Rate Decay**

  - Learning rate decay is applied in tine tuning stage

  ![learning rage in training](./resources/figure/005-learning_rate.png)

  - Learning rate is decreased by factor of 10 for every 250 epochs

- Epochs 

  - Pre-training stage: 100
  - Fine tuning stage : 600

------

## Data Set

### Training Data

&gt; IDN-TensorFlow/data/generate_dataset/train_data.m

- 291 images
  - Download from Author's Repository
- Data Augmentations (Rotation, flip) were used
- Scale Factor : $\times 2$, $\times 3$, $\times 4$
- Patch size 

| scale | Pre-Training (LR / GT) | Fine Tuning (LR / GT) |
| ----- | ---------------------- | --------------------- |
| 2     | 29 / 58                | 39 / 78               |
| 3     | 15 / 45                | 26 / 78               |
| 4     | 11 / 44                | 19 / 76               |

- Batch size : 64



### Testing Data

&gt; IDN-TensorFlow/data/generate_dataset/test_data.m

- Set5, Set14, B100, Urban100
  - Download from Author's page [[zip(test)]](https://cv.snu.ac.kr/research/VDSR/test_data.zip)
- Bicubic interpolation is used for LR data acquisition
- Scale Factor : $\times 2$, $\times 3$, $\times 4$

------

## Results

### Validation

PSNR performance plot on Set5

- Scale 2

  |              |             |
  | ------------ | ----------- |
  | Pre-training | Fine Tuning |

  

- Scale 3

  |              |             |
  | ------------ | ----------- |
  | Pre-training | Fine Tuning |

  

- Scale 4

  |              |             |
  | ------------ | ----------- |
  | Pre-training | Fine Tuning |

  

### Objective Quality Assessment

#### Methods

- Bicubic Interpolation 
  - imresize(..., ..., 'bicubic') in Matlab
- IDN(Original)
  - Author's Caffe implementation [[Code]](https://github.com/Zheng222/IDN-Caffe)
- IDN (TensorFlow)
  - TensorFlow implementation
  - Train Details for Comparison
    - Data Augmentation
      - Rotation : 90°, 180°, 270°
      - Flip : left / right
      - Down scale : 0.9, 0.8, 0.7, 0.6

#### Average PSNR/SSIM

- **Set5**
- Pre-Training

| scale      | Bicubic        | IDN (Original) | IDN (TensorFlow) |
| ---------- | -------------- | -------------- | ---------------- |
| " />\times 2<img src="https://tex.s2cms.ru/svg/%20%7C%2033.68%20%2F%200.9304%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%20" alt=" | 33.68 / 0.9304 |                |                  |
| " />\times 3<img src="https://tex.s2cms.ru/svg/%20%7C%2030.40%20%2F%200.8682%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%20" alt=" | 30.40 / 0.8682 |                |                  |
| " />\times 4<img src="https://tex.s2cms.ru/svg/%20%7C%2028.43%20%2F%200.8104%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%0A%0A%0A-%20Fine%20Tuning%0A%0A%7C%20scale%20%20%20%20%20%20%7C%20Bicubic%20%20%20%20%20%20%20%20%7C%20IDN%20(Original)%20%7C%20IDN%20(TensorFlow)%20%7C%0A%7C%20----------%20%7C%20--------------%20%7C%20--------------%20%7C%20----------------%20%7C%0A%7C%20" alt=" | 28.43 / 0.8104 |                |                  |



- Fine Tuning

| scale      | Bicubic        | IDN (Original) | IDN (TensorFlow) |
| ---------- | -------------- | -------------- | ---------------- |
| " />\times 2<img src="https://tex.s2cms.ru/svg/%20%7C%2033.68%20%2F%200.9304%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%20" alt=" | 33.68 / 0.9304 |                |                  |
| " />\times 3<img src="https://tex.s2cms.ru/svg/%20%7C%2030.40%20%2F%200.8682%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%20" alt=" | 30.40 / 0.8682 |                |                  |
| " />\times 4<img src="https://tex.s2cms.ru/svg/%20%7C%2028.43%20%2F%200.8104%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%0A%0A%0A-%20**Set14**%0A-%20Pre-Training%0A%0A%7C%20scale%20%20%20%20%20%20%7C%20Bicubic%20%20%20%20%20%20%20%20%7C%20IDN%20(Original)%20%7C%20IDN%20(TensorFlow)%20%7C%0A%7C%20----------%20%7C%20--------------%20%7C%20--------------%20%7C%20----------------%20%7C%0A%7C%20" alt=" | 28.43 / 0.8104 |                |                  |



- **Set14**
- Pre-Training

| scale      | Bicubic        | IDN (Original) | IDN (TensorFlow) |
| ---------- | -------------- | -------------- | ---------------- |
| " />\times 2<img src="https://tex.s2cms.ru/svg/%20%7C%2030.24%20%2F%200.8693%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%20" alt=" | 30.24 / 0.8693 |                |                  |
| " />\times 3<img src="https://tex.s2cms.ru/svg/%20%7C%2027.54%20%2F%200.7746%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%20" alt=" | 27.54 / 0.7746 |                |                  |
| " />\times 4<img src="https://tex.s2cms.ru/svg/%20%7C%2026.00%20%2F%200.7029%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%0A%0A%0A-%20Fine%20Tuning%0A%0A%7C%20scale%20%20%20%20%20%20%7C%20Bicubic%20%20%20%20%20%20%20%20%7C%20IDN%20(Original)%20%7C%20IDN%20(TensorFlow)%20%7C%0A%7C%20----------%20%7C%20--------------%20%7C%20--------------%20%7C%20----------------%20%7C%0A%7C%20" alt=" | 26.00 / 0.7029 |                |                  |



- Fine Tuning

| scale      | Bicubic        | IDN (Original) | IDN (TensorFlow) |
| ---------- | -------------- | -------------- | ---------------- |
| " />\times 2<img src="https://tex.s2cms.ru/svg/%20%7C%2030.24%20%2F%200.8693%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%20" alt=" | 30.24 / 0.8693 |                |                  |
| " />\times 3<img src="https://tex.s2cms.ru/svg/%20%7C%2027.54%20%2F%200.7746%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%20" alt=" | 27.54 / 0.7746 |                |                  |
| " />\times 4<img src="https://tex.s2cms.ru/svg/%20%7C%2026.00%20%2F%200.7029%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%0A-%20**B100**%0A-%20Pre-Training%0A%0A%7C%20scale%20%20%20%20%20%20%7C%20Bicubic%20%20%20%20%20%20%20%20%7C%20IDN%20(Original)%20%7C%20IDN%20(TensorFlow)%20%7C%0A%7C%20----------%20%7C%20--------------%20%7C%20--------------%20%7C%20----------------%20%7C%0A%7C%20" alt=" | 26.00 / 0.7029 |                |                  |

- **B100**
- Pre-Training

| scale      | Bicubic        | IDN (Original) | IDN (TensorFlow) |
| ---------- | -------------- | -------------- | ---------------- |
| " />\times 2<img src="https://tex.s2cms.ru/svg/%20%7C%2029.56%20%2F%200.8442%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%20" alt=" | 29.56 / 0.8442 |                |                  |
| " />\times 3<img src="https://tex.s2cms.ru/svg/%20%7C%2027.21%20%2F%200.7401%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%20" alt=" | 27.21 / 0.7401 |                |                  |
| " />\times 4<img src="https://tex.s2cms.ru/svg/%20%7C%2025.96%20%2F%200.6697%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%0A%0A%0A-%20Fine%20Tuning%0A%0A%7C%20scale%20%20%20%20%20%20%7C%20Bicubic%20%20%20%20%20%20%20%20%7C%20IDN%20(Original)%20%7C%20IDN%20(TensorFlow)%20%7C%0A%7C%20----------%20%7C%20--------------%20%7C%20--------------%20%7C%20----------------%20%7C%0A%7C%20" alt=" | 25.96 / 0.6697 |                |                  |



- Fine Tuning

| scale      | Bicubic        | IDN (Original) | IDN (TensorFlow) |
| ---------- | -------------- | -------------- | ---------------- |
| " />\times 2<img src="https://tex.s2cms.ru/svg/%20%7C%2029.56%20%2F%200.8442%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%20" alt=" | 29.56 / 0.8442 |                |                  |
| " />\times 3<img src="https://tex.s2cms.ru/svg/%20%7C%2027.21%20%2F%200.7401%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%20" alt=" | 27.21 / 0.7401 |                |                  |
| " />\times 4<img src="https://tex.s2cms.ru/svg/%20%7C%2025.96%20%2F%200.6697%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%0A-%20**Urban100**%0A-%20Pre-Training%0A%0A%7C%20scale%20%20%20%20%20%20%7C%20Bicubic%20%20%20%20%20%20%20%20%7C%20IDN%20(Original)%20%7C%20IDN%20(TensorFlow)%20%7C%0A%7C%20----------%20%7C%20--------------%20%7C%20--------------%20%7C%20----------------%20%7C%0A%7C%20" alt=" | 25.96 / 0.6697 |                |                  |

- **Urban100**
- Pre-Training

| scale      | Bicubic        | IDN (Original) | IDN (TensorFlow) |
| ---------- | -------------- | -------------- | ---------------- |
| " />\times 2<img src="https://tex.s2cms.ru/svg/%20%7C%2026.88%20%2F%200.8410%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%20" alt=" | 26.88 / 0.8410 |                |                  |
| " />\times 3<img src="https://tex.s2cms.ru/svg/%20%7C%2024.46%20%2F%200.7358%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%20" alt=" | 24.46 / 0.7358 |                |                  |
| " />\times 4<img src="https://tex.s2cms.ru/svg/%20%7C%2023.14%20%2F%200.6588%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%0A%0A%0A-%20Fine%20Tuning%0A%0A%7C%20scale%20%20%20%20%20%20%7C%20Bicubic%20%20%20%20%20%20%20%20%7C%20IDN%20(Original)%20%7C%20IDN%20(TensorFlow)%20%7C%0A%7C%20----------%20%7C%20--------------%20%7C%20--------------%20%7C%20----------------%20%7C%0A%7C%20" alt=" | 23.14 / 0.6588 |                |                  |



- Fine Tuning

| scale      | Bicubic        | IDN (Original) | IDN (TensorFlow) |
| ---------- | -------------- | -------------- | ---------------- |
| " />\times 2<img src="https://tex.s2cms.ru/svg/%20%7C%2026.88%20%2F%200.8410%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%20" alt=" | 26.88 / 0.8410 |                |                  |
| " />\times 3<img src="https://tex.s2cms.ru/svg/%20%7C%2024.46%20%2F%200.7358%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%0A%7C%20" alt=" | 24.46 / 0.7358 |                |                  |
| " />\times 4$$ | 23.14 / 0.6588 |                |                  |

------

### Visual Quality

- "img002" of Urban100 for scale factor $\times 2$

| Ground Truth | Bicubic | IDN (TensorFlow) Pre-train | IDN (Tensorflow) Fine Tuning |
| ------------ | ------- | -------------------------- | ---------------------------- |
|              |         |                            |                              |
|              |         |                            |                              |

- ???
- ???
- ???

------

## Difference with Authors Implementation

Image Size

Epochs and Learning rate decay step



------

## Usage

> **On Windows**
>
> - run.bat
>
> **On Linux** 
>
> - run.sh



### Training Command

Examples in scale 2

- in run.bat/sh

  - Pre-training

    python main.py --model_name=idn_pre_x2 --is_train=True --scale=2 --pretrain=False --epochs=100 --data_path=data/train_data/idn_train_x2.h5

    

  - Fine Tuning

    python main.py --model_name=idn_x2 --is_train=True --scale=2 --pretrain=True --pretrained_model_name=idn_pre_x2 --learning_rate_decay=True --decay_step=250 --epochs=600 --data_path=data/train_data/idn_fine_tuning_x2.h5 

If you want to change other parameters for training, please see the file

> VDSR-TensorFlow/model/configurations.py 



### Testing Command

Examples in scale 2

in run.bat/sh

python main.py --model_name=idn_pre_x2 --is_train=False --scale=2



### Trained checkpoint in experiments

- checkpoint [[download]](https://drive.google.com/file/d/1wiej51wFY0oYsoKF7gGiWZT5_t5mFt0f/view?usp=sharing)

### Training dataset

- vdsr_train.h5 [[download]](https://drive.google.com/file/d/1wiej51wFY0oYsoKF7gGiWZT5_t5mFt0f/view?usp=sharing)



------
