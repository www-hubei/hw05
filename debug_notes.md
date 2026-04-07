调试记录 (Debug Notes)

记录 1：环境库缺失
现象：执行 `import matplotlib` 报错 `ModuleNotFoundError`。
原因分析：当前 Python 环境中未预装第三方可视化库。
修改点：在命令行执行 `pip install matplotlib` 完成安装。

记录 2：MNIST 数据集自动下载
现象：首次运行代码时进度条缓慢或超时。
原因分析：国内访问 torchvision 默认数据源较慢。
修改点：保持程序运行，耐心等待数据下载完成。代码中的 `download=True` 成功在本地建立了 `./data` 目录。

记录 3：计算设备设置
现象：默认尝试使用 CUDA 导致报错。
原因分析：本地电脑没有 NVIDIA GPU 或驱动环境。
修改点：使用 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` 逻辑，确保代码在 CPU 模式下也能流畅运行。