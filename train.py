import torch
from ultralytics import YOLO

def main():
    # 创建 YOLO 实例并加载预训练模型
    model = YOLO("yolov8x-seg.pt")  # 你可以根据需要选择其他预训练模型，如 yolov8n.pt、yolov8s.pt 等

    # 确定设备
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 训练模型
    results = model.train(
        data=r"C:\Users\14168\1\Python\pythonProject\ultralytics\box\box_parameter\coco128.yaml",  # 数据集配置文件路径
        epochs=200,  # 训练的轮数
        batch=8,  # 每批次的图像数量
        imgsz=640,  # 输入图像的尺寸
        device=device,  # 使用 GPU 还是 CPU
        workers=1,  # 数据加载的工作线程数量
    )

    # 打印训练结果
    print("Training completed!")
    print("Results:", results)

if __name__ == '__main__':
    main()