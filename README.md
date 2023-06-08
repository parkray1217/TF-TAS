# Training-free Transformer Architecture Search

This project is used to update news related to our paper [***Training-free Transformer Architecture Search***](https://arxiv.org/pdf/2203.12217.pdf) (CVPR 2022).

## Getting Started

### Prerequisites

You will need [Python > 3](https://www.python.org/downloads) and the packages specified in _requirements.txt_.
We recommend setting up a [virtual environment with pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
and installing the packages there.

Install packages with:

```
$ pip install -r requirements.txt
```

### Data preparation
The layout of Imagenet data:
```bash
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
```

**For Hyperspectralformer search**

Indian pine dataset is downloaded

(simply use patches=7,band patches=3 as same as we use in spectralformer, you can modify the "datasethyper.py" to change different combination for the input)

### Searching

bash search_autoformer.sh

**For hyperspectralformer**

use "search_indian.sh"

### Retraining

**Note that the subnet is specified in train_searched_result.sh with "--cfg" **

bash train_searched_result.sh

**For hyperspectralformer**

use "train_searched_200_hyper" for multi-job, change the amount inside to fit different subnets

## Citation

If you use our code for your paper, please cite:
```bibtex
@inproceedings{zhou2022training,
  title={Training-free Transformer Architecture Search},
  author={Zhou, Qinqin and Sheng, Kekai and Zheng, Xiawu and Li, Ke and Sun, Xing and Tian, Yonghong and Chen, Jie and Ji, Rongrong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10894--10903},
  year={2022}
}
```



