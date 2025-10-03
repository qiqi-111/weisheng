from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import onnxruntime as ort
import base64
from PIL import Image
import io
import json
import os
import traceback

app = Flask(__name__)

# 允许跨域请求
from flask_cors import CORS

CORS(app)

print("正在加载YOLO模型...")
try:
    session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
    print("YOLO模型加载成功!")
except Exception as e:
    print(f"模型加载失败: {e}")
    session = None

# 类别名称（根据你的训练数据）
class_names = ['paper', 'cup', 'citrus', 'bottle', 'battery']
class_names_chinese = ['纸张', '杯子', '果皮', '瓶子', '电池']


@app.route('/')
def home():
    return jsonify({
        'message': 'YOLO垃圾检测服务运行中',
        'status': 'active',
        'model_loaded': session is not None
    })


@app.route('/detect', methods=['POST', 'GET'])
def detect():
    if request.method == 'GET':
        return jsonify({'message': '请使用POST方法上传图片'})

    try:
        # 接收JSON数据
        data = request.get_json()
        print("收到请求数据")

        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': '没有收到图片数据'})

        if session is None:
            return jsonify({'success': False, 'error': '模型未加载'})

        # 解码base64图片
        print("开始解码图片...")
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        print(f"图片格式: {image.format}, 模式: {image.mode}, 大小: {image.size}")

        # 转换为RGB（处理PNG透明通道）
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_np = np.array(image)
        print(f"图片numpy数组形状: {image_np.shape}")

        # 预处理
        print("开始预处理...")
        input_tensor = preprocess(image_np)
        print(f"输入张量形状: {input_tensor.shape}")

        # 模型推理
        print("开始模型推理...")
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
        print(f"模型输出数量: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"输出{i}形状: {output.shape}")

        # 后处理
        print("开始后处理...")
        detections = postprocess(outputs[0])
        print(f"检测到 {len(detections)} 个对象")

        return jsonify({
            'success': True,
            'detections': detections,
            'model_info': 'yolov8_onnx',
            'detection_count': len(detections)
        })

    except Exception as e:
        print(f"检测错误: {str(e)}")
        error_traceback = traceback.format_exc()
        print(f"错误堆栈: {error_traceback}")

        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': error_traceback
        })


def preprocess(image):
    """YOLO预处理"""
    try:
        print(f"预处理输入形状: {image.shape}")
        # 调整大小到640x640
        image = cv2.resize(image, (640, 640))
        print(f"调整大小后形状: {image.shape}")
        # 归一化
        image = image.astype(np.float32) / 255.0
        print(f"归一化后数据类型: {image.dtype}")
        # HWC to CHW
        image = image.transpose(2, 0, 1)
        print(f"转置后形状: {image.shape}")
        # 添加batch维度
        image = np.expand_dims(image, 0)
        print(f"添加batch维度后形状: {image.shape}")
        return image
    except Exception as e:
        print(f"预处理错误: {str(e)}")
        raise e


def postprocess(outputs, conf_threshold=0.25):
    """YOLOv8后处理 - 最终正确版本"""
    try:
        print(f"后处理输入形状: {outputs.shape}")

        # YOLOv8输出格式: [1, 9, 8400]
        # 实际格式是: [x, y, w, h, class0, class1, class2, class3, class4]
        # 注意：输出是原始logits，需要sigmoid激活

        outputs = outputs[0]  # [9, 8400]
        outputs = outputs.transpose(1, 0)  # [8400, 9]

        print(f"转换后形状: {outputs.shape}")

        # 应用sigmoid激活函数到类别分数部分
        outputs[:, 4:] = 1.0 / (1.0 + np.exp(-outputs[:, 4:]))

        detections = []

        for i in range(outputs.shape[0]):
            detection = outputs[i]

            # 前4个是边界框 [x, y, w, h] - 这些不需要sigmoid
            bbox = detection[:4]

            # 后5个是类别置信度（已经过sigmoid）
            class_scores = detection[4:]

            # 找到最高置信度的类别
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence > conf_threshold:
                x, y, w, h = bbox

                if class_id < len(class_names):
                    detections.append({
                        'class': int(class_id),
                        'name': class_names[class_id],
                        'chineseName': class_names_chinese[class_id],
                        'confidence': float(confidence),
                        'bbox': {
                            'x': float(x),
                            'y': float(y),
                            'width': float(w),
                            'height': float(h)
                        }
                    })

        print(f"初步检测到 {len(detections)} 个对象")

        # 应用NMS
        if detections:
            result = non_max_suppression(detections)
            print(f"NMS后剩余 {len(result)} 个对象")
            return result
        else:
            # 如果没检测到，尝试降低阈值
            print("未检测到对象，尝试降低阈值...")
            return postprocess_low_confidence(outputs)

    except Exception as e:
        print(f"后处理错误: {str(e)}")
        print(f"错误堆栈: {traceback.format_exc()}")
        return []


def postprocess_low_confidence(outputs, conf_threshold=0.1):
    """低置信度后处理"""
    detections = []

    for i in range(outputs.shape[0]):
        detection = outputs[i]
        class_scores = detection[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence > conf_threshold:
            x, y, w, h = detection[:4]

            if class_id < len(class_names):
                detections.append({
                    'class': int(class_id),
                    'name': class_names[class_id],
                    'chineseName': class_names_chinese[class_id],
                    'confidence': float(confidence),
                    'bbox': {
                        'x': float(x),
                        'y': float(y),
                        'width': float(w),
                        'height': float(h)
                    }
                })

    print(f"低阈值检测到 {len(detections)} 个对象")

    if detections:
        result = non_max_suppression(detections)
        print(f"低阈值NMS后剩余 {len(result)} 个对象")
        return result

    return []


def non_max_suppression(detections, iou_threshold=0.45):
    """非极大值抑制"""
    if not detections:
        return []

    # 按置信度排序
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    keep = []

    while detections:
        # 取置信度最高的检测
        best = detections.pop(0)
        keep.append(best)

        # 计算与剩余检测的IoU
        remaining = []
        for detection in detections:
            iou = calculate_iou(best['bbox'], detection['bbox'])
            if iou < iou_threshold:
                remaining.append(detection)

        detections = remaining

    return keep


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    # 将中心坐标转换为角坐标
    x1 = box1['x'] - box1['width'] / 2
    y1 = box1['y'] - box1['height'] / 2
    x2 = box1['x'] + box1['width'] / 2
    y2 = box1['y'] + box1['height'] / 2

    x3 = box2['x'] - box2['width'] / 2
    y3 = box2['y'] - box2['height'] / 2
    x4 = box2['x'] + box2['width'] / 2
    y4 = box2['y'] + box2['height'] / 2

    # 计算交集区域
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算并集区域
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)