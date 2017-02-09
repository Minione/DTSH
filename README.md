Deep Supervised Hashing with Triplet Labels 
===================

This repository contains the code for the deep hashing approach proposed in our paper: 

["Deep Supervised Hashing with Triplet Labels".] (http://www.andrew.cmu.edu/user/xiaofan2/paper/DeepHashing-ACCV16.pdf) Xiaofang Wang, Yi Shi and Kris M. Kitani. ACCV 2016.

Part of the code is modified from [here] (http://cs.nju.edu.cn/lwj/code/DPSH.zip).

## Requirements ##
This code is written in MATLAB and requires [MatConvNet] (http://www.vlfeat.org/matconvnet/).

## Preparation ##
- Download the CIFAR-10 dataset from [here] (https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz). Uncompress the file and put the folder "cifar-10-batches-mat/" under the main folder.
- Download the Pretrained VGG-F model from [here] (http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat). Put the model under the main folder.

## Usage ##
Run the following command in MATLAB:
```
$ DTSH(24)
```

Here 24 represents the traget code length and you may change it to other numbers.

## Reference ##
If you use the code, please cite:
```
@article{wang2016deep,
  title={Deep Supervised Hashing with Triplet Labels},
  author={Wang, Xiaofang and Shi, Yi and Kitani, Kris M},
  journal={Asian Conference on Computer Vision},
  year={2016}
}
```

## Contact ##
- Xiaofang Wang (Robotics Institute, Carnegie Mellon University)
- xiaofan2@andrew.cmu.edu
- [Personal Homepage] (http://www.andrew.cmu.edu/user/xiaofan2/)
