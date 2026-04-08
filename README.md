人工智能导论 - 课程作业 05
极简 CNN 手写数字识别与 LeNet-5 实现

本项目包含从基础卷积神经网络（CNN）到经典 LeNet-5 结构的完整实验流程。通过对 MNIST 数据集的训练与测试，深入理解 CNN 各组件的作用及其对识别精度的影响。



 📁 项目目录结构

 `simple_cnn.py`: 任务一核心。包含极简 CNN 的网络定义、数据自动下载及训练逻辑。
 `lenet5.py`: 任务二模型。独立实现的经典 LeNet-5 网络结构类。
 `train_lenet5.py`: 任务二脚本。专门用于 LeNet-5 模型的训练与测试。
 `report.md`: 实验报告。包含模型参数量推导表格、结构对比及实验分析。
 `debug_notes.md`: 调试笔记。记录了实验过程中遇到的报错（如 Matplotlib 字体）及修复过程。
* `training_loss.png`: 实验结果图。极简 CNN 训练过程中的 Loss 下降曲线。
* `simple_cnn.pth`: 权重文件。训练好的 SimpleCNN 模型参数。

