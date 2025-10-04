# app_pytorch.py - PyTorch版本垃圾检测服务
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
import base64
from PIL import Image
import io
import os
import traceback

app = Flask(__name__)
CORS(app)

print("正在加载YOLO PyTorch模型...")
try:
    # 使用Ultralytics YOLO加载
    from ultralytics import YOLO

    model = YOLO('best.pt')
    print("✅ YOLO PyTorch模型加载成功!")

    # 测试模型是否能正常工作
    print("进行模型健康检查...")
    test_image = np.ones((640, 640, 3), dtype=np.uint8) * 128
    test_results = model.predict(test_image, imgsz=640, conf=0.5, verbose=False)
    print("✅ 模型健康检查通过!")

except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print(f"错误详情: {traceback.format_exc()}")
    model = None

# 类别名称
class_names = ['paper', 'cup', 'citrus', 'bottle', 'battery']
class_names_chinese = ['纸张', '杯子', '果皮', '瓶子', '电池']


@app.route('/')
def home():
    return jsonify({
        'message': 'YOLO垃圾检测服务运行中(PyTorch版本)',
        'status': 'active',
        'model_loaded': model is not None
    })


@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'GET':
        return jsonify({'message': '请使用POST方法上传图片'})

    try:
        data = request.get_json()
        print("📨 收到请求数据")

        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': '没有收到图片数据'})

        if model is None:
            return jsonify({'success': False, 'error': '模型未加载'})

        # 解码图片
        print("🖼️ 开始解码图片...")
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        original_size = image.size
        print(f"图片原始尺寸: {original_size}")

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 使用YOLO模型的predict方法
        print("🔍 开始模型推理...")
        results = model.predict(
            image,
            imgsz=640,
            conf=0.3,  # 置信度阈值
            iou=0.5,  # NMS IoU阈值
            verbose=False  # 不输出详细信息
        )

        print("📊 开始后处理...")
        detections = process_yolo_results(results)
        print(f"✅ 检测完成: {len(detections)} 个对象")

        # 打印检测详情
        for i, det in enumerate(detections):
            print(f"  检测{i + 1}: {det['chineseName']} - 置信度: {det['confidence']:.3f}")

        return jsonify({
            'success': True,
            'detections': detections,
            'model_info': 'yolov8_pytorch',
            'detection_count': len(detections)
        })

    except Exception as e:
        print(f"❌ 检测错误: {str(e)}")
        error_traceback = traceback.format_exc()
        print(f"错误堆栈: {error_traceback}")

        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': error_traceback
        })


def process_yolo_results(results):
    """处理YOLO预测结果"""
    detections = []

    if not results or len(results) == 0:
        print("⚠️ 没有检测结果")
        return detections

    # 获取第一个结果（单张图片）
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print("⚠️ 没有检测到任何对象")
        return detections

    # 获取边界框信息
    boxes = result.boxes
    print(f"📦 原始检测框数量: {len(boxes)}")

    for i in range(len(boxes)):
        # 获取边界框坐标 (xyxy格式)
        box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
        confidence = boxes.conf[i].cpu().numpy()  # 置信度
        class_id = int(boxes.cls[i].cpu().numpy())  # 类别ID

        if class_id < len(class_names):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # 检查边界框是否合理
            if width > 10 and height > 10:  # 最小尺寸限制
                detections.append({
                    'class': class_id,
                    'name': class_names[class_id],
                    'chineseName': class_names_chinese[class_id],
                    'confidence': float(confidence),
                    'bbox': {
                        'x': float(x1),
                        'y': float(y1),
                        'width': float(width),
                        'height': float(height)
                    }
                })

    print(f"📏 尺寸过滤后: {len(detections)} 个对象")

    # 按置信度排序
    detections.sort(key=lambda x: x['confidence'], reverse=True)

    # 应用额外的NMS（虽然YOLO已经有NMS，但可以再加一层保险）
    final_detections = []
    while detections and len(final_detections) < 15:  # 最多15个检测
        best = detections.pop(0)
        final_detections.append(best)

        # 移除重叠的检测
        detections = [
            det for det in detections
            if calculate_iou(best['bbox'], det['bbox']) < 0.3
        ]

    print(f"🎯 NMS后最终结果: {len(final_detections)} 个对象")
    return final_detections


def calculate_iou(box1, box2):
    """计算IoU"""
    box1_x1, box1_y1 = box1['x'], box1['y']
    box1_x2, box1_y2 = box1['x'] + box1['width'], box1['y'] + box1['height']

    box2_x1, box2_y1 = box2['x'], box2['y']
    box2_x2, box2_y2 = box2['x'] + box2['width'], box2['y'] + box2['height']

    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8081))  # 使用不同端口，避免冲突
    print(f"🚀 启动PyTorch版本服务，端口: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)