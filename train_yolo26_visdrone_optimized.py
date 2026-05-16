import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from ultralytics import YOLO

if __name__ == '__main__':
    # 从模型配置路径自动提取名称
    model_cfg = 'ultralytics/cfg/models/26/yolo26.yaml'
    model_name = os.path.splitext(os.path.basename(model_cfg))[0]  # yolo26-p2 -> yolo26_visdrone_p2

    # 加载YOLO26模型（使用配置文件创建模型）
    model = YOLO(model_cfg)  # 使用配置文件创建模型

    # 训练参数配置（优化版）
    train_params = {
        'data': 'd:\\yolo_car_search(type_4)\\datasets\\visdrone.yaml',  # 数据集配置文件
        'epochs': 150,  # 训练轮数，增加以提高密集场景的性能
        'imgsz': 640,  # 输入图像大小
        'batch': 4,  # 批次大小，适合5060显卡的4GB显存
        'nbs': 8,  # 名义 batch size；与 batch=4 配合使梯度累积步数为 2，等效 batch=8 但显存仍是 4 张图
        'device': 0,  # GPU设备ID，0表示第一个GPU
        'workers': 2,  # 减少数据加载线程数，降低内存使用
        'optimizer': 'AdamW',  # 优化器，AdamW在小批量训练时表现更好
        'lr0': 0.001,  # 初始学习率，小批量时使用较小的学习率
        'momentum': 0.937,  # 动量
        'weight_decay': 0.0005,  # 权重衰减
        'warmup_epochs': 3.0,  # 预热轮数
        'warmup_momentum': 0.8,  # 预热动量
        'warmup_bias_lr': 0.1,  # 预热偏置学习率
        'box': 7.5,  # 边界框损失权重
        'cls': 0.5,  # 分类损失权重
        'dfl': 0.0,  # DFL loss 权重；reg_max=1 时 DFL 不计算，设为 0 避免误解
        'cos_lr': True,  # 余弦退火学习率调度，长训练更平稳
        'close_mosaic': 10,  # 最后 10 个 epoch 关闭 mosaic，提升小目标定位稳定性

        'augment': True,  # 启用数据增强
        'cache': False,  # 关闭缓存，减少内存使用
        'project': 'runs\\train',  # 训练结果保存目录
        'name': f'yolo26_visdrone_{model_name}_optimized',  # 训练运行名称，自动关联模型配置
        'exist_ok': False,  # 不允许覆盖现有运行，自动生成新目录
        'save_period': 15,  # 每15轮保存一次模型
        'patience': 70,  # 早停耐心值，增加以适应更长的训练
        'freeze': None,  # 冻结层
        'save_json': True,  # 保存COCO格式的JSON结果
        'save_hybrid': False,  # 保存混合模型
        'conf': 0.001,  # 置信度阈值
        'iou': 0.65,  # NMS IoU阈值，稍微降低以减少密集场景中的重叠检测
        'max_det': 500,  # 最大检测数量，增加以处理密集场景
        'half': True,  # 使用半精度训练，减少显存使用
        'dnn': False,  # 使用DNN后端
        'plots': True,  # 生成训练过程的图表
        'deterministic': False,  # Windows 下关闭 deterministic 可避免部分 CUDA 算子限制，略微提速
    }  # 类别过滤由配置文件处理

    # 开始训练
    print("开始训练YOLO26模型（优化版）...")
    train_results = model.train(**train_params)

    # 评估模型
    print("评估模型性能...")
    val_results = model.val()

    print("训练完成！")
    print(f"训练结果保存在: {train_results.save_dir}")
