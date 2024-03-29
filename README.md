# intent detection and slot filling
智能对话中的意图识别和槽位填充联合模型

## Data
- 数据来自于国外航空订票数据atis(目录atis下)。
* ![image](https://raw.githubusercontent.com/jiangnanboy/intent_detection_and_slot_filling/master/img/data_statistics.png)
* ![image](https://raw.githubusercontent.com/jiangnanboy/intent_detection_and_slot_filling/master/img/raw_data.png)

- 数据集的构建使用torchtext。[process_raw_data](https://github.com/jiangnanboy/intent_detection_and_slot_filling/blob/master/process_raw_data.ipynb) 将原始数据处理成csv结构;[build_dataset](https://github.com/jiangnanboy/intent_detection_and_slot_filling/blob/master/build_dataset.ipynb) 构建train及val数据。
* ![image](https://raw.githubusercontent.com/jiangnanboy/intent_detection_and_slot_filling/master/img/process_data.png)

- 利用apex进行混合精度训练。


## Model
 可提高训练时长，调整超参，以达到更高精度。
### model1
* ![image](https://raw.githubusercontent.com/jiangnanboy/intent_detection_and_slot_filling/master/img/model1.png)
* [Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling](https://arxiv.org/pdf/1609.01454.pdf)
* [train](https://github.com/jiangnanboy/intent_detection_and_slot_filling/blob/master/model1/train.ipynb)
* [predict](https://github.com/jiangnanboy/intent_detection_and_slot_filling/blob/master/model1/predict.ipynb)

### model2
* ![image](https://raw.githubusercontent.com/jiangnanboy/intent_detection_and_slot_filling/master/img/model2.png)
* [Attention-Based BIRNN](https://arxiv.org/pdf/1609.01454.pdf)
* [train](https://github.com/jiangnanboy/intent_detection_and_slot_filling/blob/master/model2/train.ipynb)
* [predict](https://github.com/jiangnanboy/intent_detection_and_slot_filling/blob/master/model2/predict.ipynb)

### model3
* ![image](https://raw.githubusercontent.com/jiangnanboy/intent_detection_and_slot_filling/master/img/model3.png)
* [Attention-Based Slot-gate](https://www.aclweb.org/anthology/N18-2118.pdf)
* [train](https://github.com/jiangnanboy/intent_detection_and_slot_filling/blob/master/model3/train.ipynb)
* [predict](https://github.com/jiangnanboy/intent_detection_and_slot_filling/blob/master/model3/predict.ipynb)

### model4
* ![image](https://raw.githubusercontent.com/jiangnanboy/intent_detection_and_slot_filling/master/img/model4.png)
* [CNN-Based](https://arxiv.org/pdf/1705.03122.pdf)
* [train](https://github.com/jiangnanboy/intent_detection_and_slot_filling/blob/master/model4/train.ipynb)
* [predict](https://github.com/jiangnanboy/intent_detection_and_slot_filling/blob/master/model4/predict.ipynb)

### model5
    此模型是本人在model4的基础上的改进，改进如下：
        1.只利用model4中的Encoder部分。
        2.加入了多个size的卷积，获取更多的特征，最后将这多个size的卷积进行连接。
        3.在embedding层后使用了一个多头注意力self-attention。
        4.最后将卷积后的特征和self-attention后的特征进行连接。
        
* ![image](https://raw.githubusercontent.com/jiangnanboy/intent_detection_and_slot_filling/master/img/model5.png)
* [train](https://github.com/jiangnanboy/intent_detection_and_slot_filling/blob/master/model5/train.ipynb)
* [predict](https://github.com/jiangnanboy/intent_detection_and_slot_filling/blob/master/model5/predict.ipynb)

### model6     
        note：bert用于意图识别与槽填充
* ![image](https://raw.githubusercontent.com/jiangnanboy/intent_detection_and_slot_filling/master/img/model6.png)
* [Bert](https://arxiv.org/pdf/1902.10909.pdf)
* [train](https://github.com/jiangnanboy/intent_detection_and_slot_filling/blob/master/model6/train.ipynb)
* [predict](https://github.com/jiangnanboy/intent_detection_and_slot_filling/blob/master/model6/predict.ipynb)


## Note

可加入Apex加速训练，使用Apex时导致的问题：
```
Loss整体变大，而且很不稳定。效果变差。会遇到梯度溢出。
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 32768.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 16384.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 8192.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 4096.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 2048.0
...
ZeroDivisionError: float division by zero

解决办法如下来防止出现梯度溢出：

1、apex中amp.initialize(model, optimizer, opt_level='O0')的opt_level由O2换成O1，再不行换成O0(欧零)
2、把batchsize从32调整为16会显著解决这个问题，另外在换成O0(欧0)的时候会出现内存不足的情况，减小batchsize也是有帮助的
3、减少学习率
4、增加Relu会有效保存梯度，防止梯度消失
```

## Requirements

* GPU & CUDA
* Python3.6.5
* PyTorch1.5
* torchtext0.6
* apex0.1

## References

Based on the following implementations

* https://github.com/bentrevett
* https://github.com/jiangnanboy/chatbot_chinese

## contact

如有搜索、推荐、nlp以及大数据挖掘等问题或合作，可联系我：

1、我的github项目介绍：https://github.com/jiangnanboy

2、我的博客园技术博客：https://www.cnblogs.com/little-horse/

3、我的QQ号:2229029156