# da-faster-rcnn-PyTorch
An unofficial implementation of  'Domain Adaptive Faster R-CNN for Object Detection in the Wild ’


### Preparation

#### Requirements: Python=3.6 and Pytorch=0.4.0

1. Install [Pytorch](http://pytorch.org/)

2. Our code is conducted based on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch),please setup the framework by it.

3. Download dataset
   
   - we use cityscape and cityscapes-foggy datasets respectly as source and target,the cityscapes dataset could be download [here](https://www.cityscapes-dataset.com/downloads/)  
   
   - the format of datasets is similar with VOC,you just need to split train.txt to train_s.txt and train_t.txt
   
   - you can also download the dataset [here]
   
   
   
   
### Train and Test

1.train the model,you need to download the pretrained model [vgg_caffe](https://github.com/jwyang/faster-rcnn.pytorch） which is different with pure pytorch pretrained model

2.change the dataset root path in /lib/model/utils/config.py and some dataset dir path in  /lib/datasets/cityscape.py

3 Train the model
 ```Shell
 # train cityscapes -> cityscapes-foggy
 CUDA_VISIBLE_DEVICES=GPU_ID python da_trainval_net.py --dataset cityscapes --net vgg16 --bs 1 --lr 2e-3 --lr_decay_step 6 --cuda
 
 # Test model in target domain 
 CUDA_VISIBLE_DEVICES=GPU_ID python eval/test.py --dataset cityscapes --part test_t --model_dir=# The path of your pth model --cuda
 ```
 
  Our model could arrive mAP=30.71% in target domain which is high than baseline which mAP=24.26%
