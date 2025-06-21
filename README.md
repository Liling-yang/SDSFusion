# SDSFusion

SDSFusion: A Semantic-Aware Infrared and Visible Image Fusion Network for Degraded Scenes

This is official code of "[SDSFusion: A Semantic-Aware Infrared and Visible Image Fusion Network for Degraded Scenes](https://ieeexplore.ieee.org/abstract/document/11014600)".

## Before Train

> conda env create -f SDSFusion.yaml

## Train

Firstly, you need to download the train dataset and eval dataset. Links for the **training** are: [train-google drive](https://drive.google.com/file/d/1dL-2-YU2S49yTb1NsFxrIa3rr5KYF3B9/view?usp=drive_link). Links for the **evaluation** are: [eval-google drive](https://drive.google.com/file/d/1krP1JKHH7CM103kbooNmHS8OtEtN_JdI/view?usp=drive_link).

Secondly, you need to set the *test_only* variable in the two **option.py** and one **main.py** to **False**. e.g.,
```
parser.add_argument('--test_only', action='store_true', default=False, help='set this option to test the model')
```

Thirdly, you need to run the **main_train.py** for the enhancement and **main.py** for the fusion.

During training, the corresponding **train_model** folder is used to store the training weight.

## Test

The checkpoint are stored in **pretrain**, which can ben downloaded from: [ckpt-google drive](https://drive.google.com/file/d/1Y8Z0FTTV9x5QrREnwtLelXfXT5eRJm8-/view?usp=sharing).

Set the *test_only* to be true. e.g.,
```
parser.add_argument('--test_only', action='store_true', default=True, help='set this option to test the model')
```

To get the coarse enhancement results, you can run the **main_test.py** in enhance_stage1, and they are stored in ./datasets/test/LLVIP/**vi_en-s1**.

To get the fine enhancement results, you can run the **main_test.py** in enhance_stage2, and they are stored in ./datasets/test/LLVIP/**vi_en-s2**.

To get the fused results, you need select the *stage* variable (stage1/stage2) in the **main.py** from fusion, and the fusion result will be placed in ./datasets/test/LLVIP/**If-s1** or ./datasets/test/LLVIP/**If-s2**. e.g.,
```
parser.add_argument('--stage', type=str, default='stage1') # or stage2
```

## The Environment

>numpy=1.15.0
>
>opencv-python=4.1.0.25
>
>python=3.7.0
>
>torch=1.8.0
>
>torchvision=0.9.0


## If this work is helpful to you, please cite it as：
```bibtex
@article{11014600,
  title={SDSFusion: A Semantic-Aware Infrared and Visible Image Fusion Network for Degraded Scenes},
  author={Chen, Jun and Yang, Liling and Yu, Wei and Gong, Wenping and Cai, Zhanchuan and Ma, Jiayi},
  journal={IEEE Transactions on Image Processing}, 
  volume={34},
  pages={3139-3153},
  year={2025}
}
