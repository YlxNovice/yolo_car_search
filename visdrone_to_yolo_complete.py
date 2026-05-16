# visdrone_to_yolo_vehicles.py
"""
VisDrone 数据集转 YOLO 格式完整脚本
只保留车辆类别：car(4), van(5), truck(6), bus(9)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter


class VisDroneVehiclesToYOLO:
    def __init__(self, visdrone_root, output_root):
        """
        初始化转换器 - 只转换车辆类别

        Args:
            visdrone_root: VisDrone数据集根目录
            output_root: YOLO格式输出目录
        """
        self.visdrone_root = Path(visdrone_root)
        self.output_root = Path(output_root)

        # 只保留车辆类别：car(4), van(5), truck(6), bus(9)
        self.target_classes = {4, 5, 6, 9}

        # 类别重新映射：VisDrone ID -> YOLO ID (0-3)
        self.class_remapping = {
            4: 0,  # car -> 0
            5: 1,  # van -> 1
            6: 2,  # truck -> 2
            9: 3,  # bus -> 3
        }

        # 类别名称
        self.class_names = ['car', 'van', 'truck', 'bus']

        # 统计信息
        self.stats = {
            'train': Counter(),
            'val': Counter(),
            'test': Counter()
        }

        # 创建YOLO目录结构
        self.create_directory_structure()

    def create_directory_structure(self):
        """创建YOLO格式的目录结构"""
        for split in ['train', 'val', 'test']:
            (self.output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    def convert_annotation(self, ann_path, img_width, img_height, split):
        """
        转换单个标注文件为YOLO格式 - 只保留车辆类别

        Args:
            ann_path: 标注文件路径
            img_width: 图片宽度
            img_height: 图片高度
            split: 数据集划分（用于统计）

        Returns:
            list: YOLO格式的标注列表
        """
        yolo_annotations = []

        try:
            with open(ann_path, 'r') as f:
                lines = f.readlines()

            # 跳过第一行（VisDrone标注文件可能有标题）
            start_idx = 1 if lines and 'bbox_left' in lines[0] else 0

            for line in lines[start_idx:]:
                parts = line.strip().split(',')
                if len(parts) < 6:  # 至少需要6个值
                    continue

                try:
                    bbox_left = float(parts[0])
                    bbox_top = float(parts[1])
                    bbox_width = float(parts[2])
                    bbox_height = float(parts[3])
                    category = int(parts[5].strip())
                except (ValueError, IndexError):
                    continue

                # 只保留目标车辆类别
                if category not in self.target_classes:
                    continue

                # 重新映射类别ID
                new_class_id = self.class_remapping[category]

                # 转换为YOLO格式（归一化坐标）
                x_center = (bbox_left + bbox_width / 2) / img_width
                y_center = (bbox_top + bbox_height / 2) / img_height
                width = bbox_width / img_width
                height = bbox_height / img_height

                # 确保坐标在[0,1]范围内
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))

                # 过滤掉太小的目标（小于2像素）
                if width * img_width < 2 or height * img_height < 2:
                    continue

                yolo_annotations.append([new_class_id, x_center, y_center, width, height])
                self.stats[split][self.class_names[new_class_id]] += 1

        except Exception as e:
            print(f"❌ 处理标注文件失败 {ann_path}: {e}")

        return yolo_annotations

    def process_single_image(self, img_path, ann_path, split):
        """
        处理单张图片和标注

        Args:
            img_path: 图片路径
            ann_path: 标注路径
            split: 数据集划分（train/val/test）
        """
        try:
            # 读取图片获取尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️ 无法读取图片: {img_path}")
                return False

            img_height, img_width = img.shape[:2]

            # 转换标注
            yolo_anns = self.convert_annotation(ann_path, img_width, img_height, split)

            # 确定输出路径
            out_img_path = self.output_root / 'images' / split / img_path.name
            out_ann_path = self.output_root / 'labels' / split / f"{img_path.stem}.txt"

            # 复制图片
            shutil.copy2(img_path, out_img_path)

            # 保存YOLO格式标注
            if yolo_anns:
                with open(out_ann_path, 'w') as f:
                    for ann in yolo_anns:
                        f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
            else:
                # 创建空文件（表示该图片没有车辆目标）
                out_ann_path.touch()

            return True

        except Exception as e:
            print(f"❌ 处理失败 {img_path}: {e}")
            return False

    def process_split(self, split, max_workers=4):
        """
        处理单个数据集划分

        Args:
            split: 数据集划分（train/val/test）
            max_workers: 并行处理的线程数
        """
        print(f"\n{'=' * 60}")
        print(f"📦 处理 {split.upper()} 数据集")
        print(f"{'=' * 60}")

        # VisDrone目录结构
        img_dir = self.visdrone_root / f'VisDrone2019-DET-{split}' / 'images'
        ann_dir = self.visdrone_root / f'VisDrone2019-DET-{split}' / 'annotations'

        # 尝试其他可能的目录结构
        if not img_dir.exists():
            img_dir = self.visdrone_root / f'VisDrone2019-DET-{split}-dev' / 'images'
            ann_dir = self.visdrone_root / f'VisDrone2019-DET-{split}-dev' / 'annotations'

        if not img_dir.exists():
            print(f"⚠️ 目录不存在: {img_dir}")
            return 0

        # 获取所有图片
        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        print(f"📷 找到 {len(img_files)} 张图片")

        # 准备任务列表
        tasks = []
        skipped = 0
        for img_path in img_files:
            ann_path = ann_dir / f"{img_path.stem}.txt"
            if not ann_path.exists():
                skipped += 1
                continue
            tasks.append((img_path, ann_path, split))

        if skipped > 0:
            print(f"⚠️ 跳过 {skipped} 张缺少标注的图片")

        print(f"🔄 开始转换 {len(tasks)} 张图片...")

        # 并行处理
        success_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_single_image, *task): task for task in tasks}

            with tqdm(total=len(tasks), desc=f"转换进度", ncols=100) as pbar:
                for future in as_completed(futures):
                    if future.result():
                        success_count += 1
                    pbar.update(1)

        print(f"✅ {split.upper()} 完成: {success_count}/{len(tasks)} 张图片")
        return success_count

    def create_dataset_config(self):
        """创建YOLO数据集配置文件"""
        config_content = f"""# VisDrone Vehicles Dataset Configuration
# 只保留车辆类别: car, van, truck, bus

# 数据集根目录
path: {self.output_root.absolute()}

# 数据集划分
train: images/train
val: images/val
test: images/test

# 类别信息
nc: 4
names: ['car', 'van', 'truck', 'bus']

# 原始VisDrone类别映射:
# 4 -> 0: car
# 5 -> 1: van
# 6 -> 2: truck
# 9 -> 3: bus
"""

        config_path = self.output_root / 'visdrone_vehicles.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)

        print(f"\n📝 配置文件已保存: {config_path}")
        return config_path

    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("📊 数据集统计信息")
        print("=" * 60)

        total_vehicles = 0

        for split in ['train', 'val', 'test']:
            if self.stats[split]:
                print(f"\n{split.upper()} 车辆目标统计:")
                split_total = 0
                for class_name in self.class_names:
                    count = self.stats[split][class_name]
                    if count > 0:
                        print(f"   {class_name:8s}: {count:8d}")
                        split_total += count
                print(f"   {'─' * 22}")
                print(f"   小计      : {split_total:8d}")
                total_vehicles += split_total

        print(f"\n{'=' * 60}")
        print(f"🎯 总计车辆目标: {total_vehicles}")
        print("=" * 60)

    def check_output_dataset(self):
        """检查输出数据集的完整性"""
        print("\n" + "=" * 60)
        print("🔍 检查输出数据集")
        print("=" * 60)

        total_images = 0
        total_vehicles = 0
        total_backgrounds = 0

        for split in ['train', 'val', 'test']:
            img_dir = self.output_root / 'images' / split
            lbl_dir = self.output_root / 'labels' / split

            if not img_dir.exists():
                continue

            images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            labels = list(lbl_dir.glob('*.txt')) if lbl_dir.exists() else []

            # 统计有目标的图片
            images_with_objects = 0
            objects_count = 0

            for lbl in labels:
                try:
                    with open(lbl, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            images_with_objects += 1
                            objects_count += len(lines)
                except:
                    pass

            background_images = len(images) - images_with_objects

            print(f"\n{split.upper()}:")
            print(f"   总图片     : {len(images)}")
            print(f"   有车辆图片 : {images_with_objects}")
            print(f"   无车辆图片 : {background_images}")
            print(f"   车辆目标数 : {objects_count}")

            # 检查配对情况
            img_names = {p.stem for p in images}
            lbl_names = {p.stem for p in labels}
            missing_labels = img_names - lbl_names

            if missing_labels:
                print(f"   ⚠️ 缺少标签   : {len(missing_labels)} 个")

            total_images += len(images)
            total_vehicles += objects_count
            total_backgrounds += background_images

        print(f"\n{'=' * 60}")
        print(f"📊 总计:")
        print(f"   图片总数   : {total_images}")
        print(f"   车辆目标数 : {total_vehicles}")
        print(f"   背景图片数 : {total_backgrounds}")
        print("=" * 60)

    def convert_all(self, max_workers=4):
        """执行完整转换"""
        print("=" * 60)
        print("🚀 VisDrone 车辆数据集转换工具")
        print("=" * 60)
        print(f"目标类别: {', '.join(self.class_names)}")
        print(f"输入目录: {self.visdrone_root.absolute()}")
        print(f"输出目录: {self.output_root.absolute()}")
        print("=" * 60)

        # 转换各个数据集划分
        splits = ['train', 'val', 'test']
        for split in splits:
            self.process_split(split, max_workers)

        # 创建配置文件
        config_path = self.create_dataset_config()

        # 打印统计信息
        self.print_statistics()

        # 检查输出
        self.check_output_dataset()

        print("\n" + "=" * 60)
        print("✨ 转换完成！")
        print("=" * 60)
        print(f"\n📋 训练命令:")
        print(f"   from ultralytics import YOLO")
        print(f"   model = YOLO('yolo26n.pt')")
        print(f"   model.train(data='{config_path}', epochs=150, imgsz=640, batch=8)")
        print("\n")


def main():
    parser = argparse.ArgumentParser(description='VisDrone转YOLO格式 - 只保留车辆类别')
    parser.add_argument('--visdrone_root', type=str, required=True,
                        help='VisDrone数据集根目录')
    parser.add_argument('--output_root', type=str, default='./datasets_vehicles',
                        help='YOLO格式输出目录')
    parser.add_argument('--workers', type=int, default=4,
                        help='并行处理线程数')

    args = parser.parse_args()

    converter = VisDroneVehiclesToYOLO(
        visdrone_root=args.visdrone_root,
        output_root=args.output_root
    )

    converter.convert_all(max_workers=args.workers)


if __name__ == '__main__':
    # 方式1: 命令行运行
    # main()

    # 方式2: 直接指定路径运行（修改为你的实际路径）
    converter = VisDroneVehiclesToYOLO(
        visdrone_root=r"D:\data",  # 你的VisDrone数据集根目录
        output_root=r"D:\yolo_car_search\datasets"  # 输出目录
    )

    converter.convert_all(max_workers=4)