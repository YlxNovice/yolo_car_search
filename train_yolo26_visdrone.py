import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO

if __name__ == '__main__':
    # 从模型配置路径自动提取名称
    model_cfg = 'ultralytics/cfg/models/26/yolo26-p2-ema.yaml'
    model_name = os.path.splitext(os.path.basename(model_cfg))[0]  # yolo26-p2 -> yolo26_visdrone_p2

    # 加载YOLO26模型（使用配置文件创建模型）
    model = YOLO(model_cfg)  # 使用配置文件创建模型

    # 训练参数配置
    train_params = {
        'data': 'd:\\yolo_car_search(type_4)\\datasets\\visdrone.yaml',  # 数据集配置文件
        'epochs': 150,  # 训练轮数，增加以提高密集场景的性能
        'imgsz': 640,  # 输入图像大小
        'batch': 4 ,  # 批次大小，适合5060显卡的4GB显存
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
        'dfl': 1.5,  # 分布 focal损失权重

        'augment': True,  # 启用数据增强
        'cache': False,  # 关闭缓存，减少内存使用
        'project': 'runs\\train',  # 训练结果保存目录
        'name': f'yolo26_visdrone_{model_name}',  # 训练运行名称，自动关联模型配置
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
    }  # 类别过滤由配置文件处理


    # 开始训练
    print("开始训练YOLO26模型...")
    train_results = model.train(**train_params)

    # 评估模型
    print("评估模型性能...")
    val_results = model.val()

    print("训练完成！")
    print(f"训练结果保存在: {train_results.save_dir}")