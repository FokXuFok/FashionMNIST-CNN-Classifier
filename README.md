# FashionMNIST-CNN-Classifier

本项目是用于大学人工智能应用于开发课程结课作品和人工智能初级学习，内容是基于5层CNN的FashionMNIST服装分类系统，采用标签平滑损失和AdamW优化器训练，提供tkinter GUI识别应用，支持单张和批量图片识别。

## 项目结构

```
├── 图像分类模型_原码.py   # 模型训练代码
├── application.py                # GUI识别应用
├── 5layer_cnn_labelsmooth_best_model.pth    # 最佳模型权重
├── 5layer_cnn_labelsmooth_final_model.pth   # 最终模型权重
├── training_history.png          # 训练曲线图
├── requirements.xlsx             # 依赖库说明
├── requirements.txt              # 依赖库列表
├── .gitignore
├── 图片/                         # 测试图片
└── data/                         # 数据集（运行时自动下载）
```

## 环境要求

| 库名 | 用途 | 安装命令 |
|------|------|----------|
| torch | 深度学习框架 | `pip install torch` |
| torchvision | 数据集和图像预处理 | `pip install torchvision` |
| Pillow | 图片读取 | `pip install Pillow` |
| matplotlib | 训练曲线可视化 | `pip install matplotlib` |
| tkinter | GUI窗口 | Python自带，无需安装 |

一键安装：
```bash
pip install torch torchvision Pillow matplotlib
```

## 使用方法

### 训练模型

```bash
python 图像分类模型_原码.py
```

训练完成后会自动保存模型权重和训练曲线图。

### 运行GUI识别应用

```bash
python application.py
```

启动后可以：
- 点击"选择图片"选取本地图片
- 点击"开始识别"对选中图片进行分类
- 点击"批量识别"自动识别"图片"文件夹下所有图片

## 模型说明

- **网络结构**：5层卷积神经网络（CNN），每层包含卷积、批归一化、ReLU激活、池化
- **损失函数**：标签平滑交叉熵损失（smoothing=0.1）
- **优化器**：AdamW（lr=0.001, weight_decay=1e-3）
- **训练轮数**：10 epochs
- **分类类别**：T恤/上衣、裤子、套头衫、连衣裙、外套、凉鞋、衬衫、运动鞋、包、短靴

## 数据集

使用 FashionMNIST 数据集，运行训练代码时会自动下载到 `data/` 目录。
