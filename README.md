# da-faster-rcnn-PyTorch
An unofficial implementation of  'Domain Adaptive Faster R-CNN for Object Detection in the Wild ’


### Preparation

#### Requirements: Python=2.7/3.6 and Pytorch=0.4.0

1. Install [Pytorch](http://pytorch.org/)

   - pip install https://download.pytorch.org/whl/cu90/torch-0.4.0-cp27-cp27m-linux_x86_64.whl

2. Our code is conducted based on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch),please setup the framework by it.

3. Download dataset
   
   - we use cityscape and cityscapes-foggy datasets respectly as source and target,the cityscapes dataset could be download [Here](https://www.cityscapes-dataset.com/downloads/)  
   
   - the format of datasets is similar with VOC,you just need to split train.txt to train_s.txt and train_t.txt
   
   - you can also download the dataset [GoogleDrive](https://drive.google.com/file/d/1mA0L5-1U_Vo-S8-cv12QBmhgG9FFf6nf/view?usp=sharing)
   
   
   
   
### Train and Test

1.train the model,you need to download the pretrained model [vgg_caffe](https://github.com/jwyang/faster-rcnn.pytorch） which is different with pure pytorch pretrained model

2.change the dataset root path in ./lib/model/utils/config.py and some dataset dir path in  ./lib/datasets/cityscape.py,the default dataset path is ./data and the default pre-trained module path is /data/ztc/detectionModel/

3 Train the model
 ```Shell
 # train cityscapes -> cityscapes-foggy
 CUDA_VISIBLE_DEVICES=GPU_ID python da_trainval_net.py --dataset cityscape --net vgg16 --bs 1 --lr 2e-3 --lr_decay_step 6 --cuda
 
 # Test model in target domain 
 CUDA_VISIBLE_DEVICES=GPU_ID python eval/test.py --dataset cityscapes --part test_t --cuda --model_dir=# The path of your pth model 
 ```
 
  Our model could arrive mAP=30.71% in target domain which is high than baseline mAP=24.26%
