import os
import sys
import traceback
import logging
from datetime import datetime
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for faster matmul
torch.backends.cudnn.allow_tf32 = True

from ultralytics import YOLO
from train_logs import generate_train_log, _get_device_name, _parse_dataset_yaml

# ========== 实时日志配置 ==========
LOGS_DIR = Path('logs')
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# 实时日志格式：时间 | 级别 | 消息
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),   # 输出到控制台
        logging.FileHandler(LOGS_DIR / 'training_realtime.log', encoding='utf-8'),  # 输出到文件
    ]
)
logger = logging.getLogger(__name__)


def _write_crash_status(log_path: Path, status: str, extra: str = ''):
    """写入崩溃恢复日志（始终追加）"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{timestamp}] 状态: {status}'
    if extra:
        line += f' | {extra}'
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


if __name__ == '__main__':
    # 从模型配置路径自动提取名称
    model_cfg = 'ultralytics/cfg/models/26/yolo26-p2-dysample-ema.yaml'
    model_name = os.path.splitext(os.path.basename(model_cfg))[0]

    # ===== 崩溃恢复日志 =====
    crash_log_path = LOGS_DIR / f'crash_recovery_{model_name}.txt'
    _write_crash_status(crash_log_path, '脚本启动', f'PID={os.getpid()}, 配置={model_cfg}')

    start_time = datetime.now()

    try:
        # ---- 阶段1：加载模型 ----
        logger.info('=' * 60)
        logger.info(f'开始训练 | 模型: {model_name} | PID: {os.getpid()}')
        logger.info('=' * 60)
        logger.info('阶段1/4: 加载模型...')
        _write_crash_status(crash_log_path, '加载模型中')

        model = YOLO(model_cfg)
        _write_crash_status(crash_log_path, '模型加载完成')

        # ---- 阶段2：配置训练参数 ----
        logger.info('阶段2/4: 配置训练参数...')
        train_params = {
            'data': 'd:\\yolo_car_search(type_4)\\datasets\\visdrone.yaml',
            'epochs': 150,
            'imgsz': 640,
            'batch': 4,
            'nbs': 8,
            'device': 0,
            'workers': 2,
            #'optimizer': 'AdamW',
            'optimizer': 'SGD',
            'lr0': 0.01,  # SGD 标准初始学习率，约 AdamW 的 10 倍
            #'lr0': 0.0005, #AdamW 标准初始学习率
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 0.0,
            'cos_lr': True,
            'close_mosaic': 10,
            'augment': True,
            'cache': False,
            'project': 'runs\\train',
            'name': f'yolo26_visdrone_{model_name}_optimized(SGD优化器)',
            'exist_ok': False,
            'save_period': 15,
            'patience': 70,
            'freeze': None,
            'save_json': True,
            'save_hybrid': False,
            'conf': 0.001,
            'iou': 0.65,
            'max_det': 500,
            'amp': True,
            'half': False,
            'dnn': False,
            'plots': True,
            'deterministic': False,
        }
        logger.info(f'训练参数: epochs={train_params["epochs"]}, batch={train_params["batch"]}, '
                     f'optimizer={train_params["optimizer"]}, imgsz={train_params["imgsz"]}')
        _write_crash_status(crash_log_path, '训练参数配置完成',
                            f'epochs={train_params["epochs"]}, batch={train_params["batch"]}')

        # ---- 阶段3：开始训练 ----
        logger.info('阶段3/4: 开始训练...')
        _write_crash_status(crash_log_path, '训练开始')
        train_results = model.train(**train_params)
        logger.info('训练阶段完成 ✓')
        _write_crash_status(crash_log_path, '训练完成')

        # ---- 阶段4：评估模型 ----
        logger.info('阶段4/4: 评估模型性能...')
        _write_crash_status(crash_log_path, '开始评估')
        val_results = model.val()
        logger.info('评估完成 ✓')
        _write_crash_status(crash_log_path, '评估完成')

        end_time = datetime.now()
        elapsed_min = (end_time - start_time).total_seconds() / 60
        logger.info(f'训练+评估总耗时: {elapsed_min:.1f} 分钟')
        logger.info(f'结果保存在: {train_results.save_dir}')

        # ---- 生成训练日志 ----
        logger.info('生成训练日志...')
        _write_crash_status(crash_log_path, '生成训练日志中')

        # 提取指标
        results_dict = getattr(val_results, 'results_dict', {}) or {}
        class_metrics = {}
        try:
            box = val_results.box
            if box is not None:
                dataset_path = Path('datasets/visdrone.yaml')
                classes, nc, _ = _parse_dataset_yaml(dataset_path)
                for i in range(nc):
                    try:
                        cp, cr, cm50, cm5095 = box.class_result(i)
                        class_metrics[i] = (cp, cr, cm50, cm5095)
                    except (IndexError, ValueError, AttributeError):
                        break
        except Exception:
            pass

        # 提取模型架构信息
        model_info = {'config_path': model_cfg}
        try:
            info = model.model.info(detailed=False, verbose=False)
            if len(info) >= 1: model_info['layers'] = int(info[0])
            if len(info) >= 2: model_info['params'] = int(info[1])
            if len(info) >= 3: model_info['gradients'] = int(info[2])
            if len(info) >= 4: model_info['gflops'] = float(info[3])
        except Exception:
            pass

        # 生成日志
        generate_train_log(
            run_dir=str(train_results.save_dir),
            model_name=model_name,
            train_params=train_params,
            metrics=results_dict,
            class_metrics=class_metrics,
            model_info=model_info,
            dataset_info=_parse_dataset_yaml(Path(train_params['data'])),
            start_time=start_time,
            end_time=end_time,
            device_name=_get_device_name(),
        )

        # ---- 全部完成，清理崩溃日志 ----
        _write_crash_status(crash_log_path, '全部完成')
        logger.info('全部完成！')
        logger.info('=' * 60)

        # 成功结束后删除崩溃恢复日志（避免残留混淆）
        if crash_log_path.exists():
            crash_log_path.unlink()

    except KeyboardInterrupt:
        logger.warning('训练被用户中断 (Ctrl+C)')
        _write_crash_status(crash_log_path, '用户中断 (Ctrl+C)',
                            f'已运行 {(datetime.now() - start_time).total_seconds() / 60:.1f} 分钟')
        sys.exit(1)

    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        error_tb = traceback.format_exc()

        # 写入崩溃日志
        _write_crash_status(crash_log_path, '崩溃',
                            f'已运行 {elapsed:.1f} 分钟')
        with open(crash_log_path, 'a', encoding='utf-8') as f:
            f.write(f'[异常类型] {type(e).__name__}\n')
            f.write(f'[错误信息] {str(e)}\n')
            f.write(f'[详细堆栈]\n{error_tb}\n')

        # 控制台输出（带分隔线方便定位）
        logger.error('=' * 60)
        logger.error(f'训练崩溃！已运行 {elapsed:.1f} 分钟')
        logger.error(f'异常类型: {type(e).__name__}')
        logger.error(f'错误信息: {str(e)}')
        logger.error(f'崩溃日志已保存至: {crash_log_path}')
        logger.error('详细堆栈:\n' + error_tb)
        logger.error('=' * 60)

        # 关键提示
        if 'CUDA' in str(e) or 'cuda' in str(e):
            logger.error('💡 提示: 疑似CUDA/显存相关错误，可尝试减小batch或imgsz')
        elif 'out of memory' in str(e).lower():
            logger.error('💡 提示: 显存不足(OOM)，可尝试减小batch、imgsz或关闭half')
        elif 'No such file' in str(e) or 'cannot find' in str(e).lower():
            logger.error('💡 提示: 文件路径错误，请检查数据集路径配置')

        sys.exit(1)
