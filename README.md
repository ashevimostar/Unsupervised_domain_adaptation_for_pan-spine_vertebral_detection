# 可学习负样本

这是《可学习负样本》的实现代码。该论文提出了一种基于负样本生成的有监督对比学习方法。

## 目录
- [摘要](#摘要)
- [方法概述](#方法概述)
- [环境配置](#环境配置)
- [数据集准备](#数据集准备)
- [使用方法](#使用方法)
  - [预训练](#预训练)
  - [微调](#微调)
- [代码结构](#代码结构)

## 摘要
...

## 方法概述
...

## 环境配置
参见 requirements.txt


## 数据集准备

详细说明实验中使用的数据集及其获取和预处理方法：

dataset ISIC 2018 
存放路径：data/data_ISIC

dataset aptos 2019
存放路径：data/data_aptos

## 使用方法

### 预训练
使用如下命令进行模型预训练：

python main_LNCo.py
或
python main_LNCo.py \
  --cls_num [分类类别数] \
  --root_path [数据存放路径]
  --csv_file_path [训练验证集数据列表文件路径] \
  --which_split [训练验证集数据列表文件路径] \
  --desc [描述信息] \
  --batch_size [批次大小] \
  --epochs [训练轮数] \
  --learning_rate [学习率] \
  --...

详细参数说明可查看代码中hpra.py或直接运行python main_pretrain.py

### 分类训练
使用预训练好的模型再单独进行分类学习微调：

bash
python main_LNCo_CL.py

## 代码结构
简要说明代码的主要结构和各文件的功能：

plaintext
project/
├── main_LNCo.py     # 训练主程序
├── main_LNCo_CL.py     # 微调主程序
├── hpra.py         # 超参文件
├── loss/              # 损失函数
│   ├── SupConLoss.py      # 引入可学习负样本的对比学习损失函数
│   ...  # 其他相关损失函数
├── data/                # 数据集
│   ├── data_ISIC      # ISIC数据集
│   ├── data_aptos    # aptos数据集
│   ...
├── dataloaders                # 数据读取
│   ├── dataset.py                  # 数据集初始化代码（该代码中会对数据集做预分析，例如为图片建立索引等，以便后续可学习负样本的使用。更换数据集时，建议调整该部分代码）（更换数据集时请调整该代码中的N_CLASSES变量，该变量表示分类类别数）
│   ├── util.py  
├── networks/               # 网络结构
│   ├── LN.py        # 可学习负样本矩阵（核心代码）
│   ├── resnet_big.py       # 主干网络
├── main_LNCo_CL_test.py     # 模型测试
├── inference.py     # 模型单独推理
├── hpra_test.py         # 推理/测试时所用超参
...
