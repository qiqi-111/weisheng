from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import onnxruntime as ort
import base64
from PIL import Image
import io
import json
import os

app = Flask(__name__)

# 允许跨域请求
from flask_cors import CORS

CORS(app)

print("正在加载YOLO模型...")
try:
    # 修复：添加 providers 参数
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
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': '没有收到图片数据'})

        if session is None:
            return jsonify({'success': False, 'error': '模型未加载'})

        # 解码base64图片
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))

        # 转换为RGB（处理PNG透明通道）
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = np.array(image)

        # 预处理
        input_tensor = preprocess(image)

        # 模型推理
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})

        # 后处理
        detections = postprocess(outputs[0])

        return jsonify({
            'success': True,
            'detections': detections,
            'model_info': 'yolov5_onnx',
            'detection_count': len(detections)
        })

    except Exception as e:
        print(f"检测错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


def preprocess(image):
    """YOLO预处理"""
    # 调整大小到640x640
    image = cv2.resize(image, (640, 640))
    # 已经是RGB，不需要转换
    # 归一化
    image = image.astype(np.float32) / 255.0
    # HWC to CHW
    image = image.transpose(2, 0, 1)
    # 添加batch维度
    image = np.expand_dims(image, 0)
    return image


def postprocess(outputs, conf_threshold=0.25):
    """YOLO后处理 - 简化版本"""
    detections = []

    # outputs[0] 形状是 [1, 25200, 85]
    # 85 = [x, y, w, h, conf, class0, class1, ... class4]

    for i in range(outputs[0].shape[1]):
        detection = outputs[0][0][i]
        confidence = detection[4]  # 物体置信度

        if confidence > conf_threshold:
            # 找到类别
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]

            # 综合置信度
            final_confidence = confidence * class_confidence

            if final_confidence > conf_threshold:
                # 提取边界框 [x, y, w, h] - 已经是中心坐标和宽高
                x, y, w, h = detection[0], detection[1], detection[2], detection[3]

                detections.append({
                    'class': int(class_id),
                    'name': class_names[class_id],
                    'chineseName': class_names_chinese[class_id],
                    'confidence': float(final_confidence),
                    'bbox': {
                        'x': float(x),
                        'y': float(y),
                        'width': float(w),
                        'height': float(h)
                    }
                })

    # 简单的NMS
    return simple_nms(detections)


def simple_nms(detections, iou_threshold=0.5):
    """简化的非极大值抑制"""
    if len(detections) == 0:
        return []

    # 按置信度排序
    detections.sort(key=lambda x: x['confidence'], reverse=True)

    keep = []
    while detections:
        current = detections.pop(0)
        keep.append(current)

        # 移除重叠的检测
        detections = [
            det for det in detections
            if calculate_iou(current['bbox'], det['bbox']) < iou_threshold
        ]

    return keep


def calculate_iou(box1, box2):
    """计算IoU"""
    # 转换为 [x1, y1, x2, y2] 格式
    box1_x1 = box1['x'] - box1['width'] / 2
    box1_y1 = box1['y'] - box1['height'] / 2
    box1_x2 = box1['x'] + box1['width'] / 2
    box1_y2 = box1['y'] + box1['height'] / 2

    box2_x1 = box2['x'] - box2['width'] / 2
    box2_y1 = box2['y'] - box2['height'] / 2
    box2_x2 = box2['x'] + box2['width'] / 2
    box2_y2 = box2['y'] + box2['height'] / 2

    # 计算交集
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
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)  # debug=False 生产环境