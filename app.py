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
        original_size = image.size  # 保存原始尺寸
        print(f"图片格式: {image.format}, 模式: {image.mode}, 大小: {original_size}")

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
        detections = postprocess(outputs[0], original_size)
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


def postprocess(outputs, original_size, conf_threshold=0.5):  # 提高置信度阈值
    """YOLOv8后处理 - 提高检测质量"""
    try:
        print(f"后处理输入形状: {outputs.shape}")
        print(f"原始图片尺寸: {original_size}")

        # YOLOv8输出格式: [1, 9, 8400]
        outputs = outputs[0]  # [9, 8400]
        outputs = outputs.transpose(1, 0)  # [8400, 9]

        print(f"转换后形状: {outputs.shape}")

        # 应用sigmoid激活函数到类别分数部分
        outputs[:, 4:] = 1.0 / (1.0 + np.exp(-outputs[:, 4:]))

        detections = []
        original_width, original_height = original_size

        # 统计每个类别的检测数量
        class_counts = {class_id: 0 for class_id in range(len(class_names))}

        for i in range(outputs.shape[0]):
            detection = outputs[i]

            # 边界框 [x, y, w, h] - 这些是归一化坐标
            bbox = detection[:4]

            # 类别置信度（已经过sigmoid）
            class_scores = detection[4:]

            # 找到最高置信度的类别
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            # 使用更高的置信度阈值来减少误检
            if confidence > conf_threshold:
                # YOLO输出的是归一化的中心坐标 [cx, cy, w, h]
                cx, cy, w, h = bbox

                # 转换为绝对像素坐标
                x_center = cx * original_width
                y_center = cy * original_height
                bbox_width = w * original_width
                bbox_height = h * original_height

                # 转换为左上角坐标
                x1 = x_center - bbox_width / 2
                y1 = y_center - bbox_height / 2

                # 检查边界框是否合理（不能太小）
                if bbox_width > 10 and bbox_height > 10:  # 最小尺寸限制
                    if class_id < len(class_names):
                        class_counts[class_id] += 1
                        detections.append({
                            'class': int(class_id),
                            'name': class_names[class_id],
                            'chineseName': class_names_chinese[class_id],
                            'confidence': float(confidence),
                            'bbox': {
                                'x': float(x1),  # 左上角x
                                'y': float(y1),  # 左上角y
                                'width': float(bbox_width),
                                'height': float(bbox_height)
                            }
                        })

        print(f"初步检测到 {len(detections)} 个对象")
        print(f"各类别检测数量: {class_counts}")

        # 应用更严格的NMS
        if detections:
            result = non_max_suppression(detections, iou_threshold=0.3)  # 更严格的IoU阈值
            print(f"严格NMS后剩余 {len(result)} 个对象")

            # 进一步过滤：只保留置信度最高的几个检测
            if len(result) > 10:  # 如果还是太多，只保留置信度最高的10个
                result = sorted(result, key=lambda x: x['confidence'], reverse=True)[:10]
                print(f"限制返回数量为10个最高置信度对象")

            # 打印最终检测结果
            final_counts = {}
            for det in result:
                class_id = det['class']
                final_counts[class_id] = final_counts.get(class_id, 0) + 1
            print(f"最终检测结果: {final_counts}")

            return result
        else:
            print("未检测到任何高置信度对象")
            return []

    except Exception as e:
        print(f"后处理错误: {str(e)}")
        print(f"错误堆栈: {traceback.format_exc()}")
        return []


def non_max_suppression(detections, iou_threshold=0.3):  # 更严格的IoU阈值
    """非极大值抑制 - 更严格版本"""
    if not detections:
        return []

    # 按置信度排序
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    keep = []

    while detections:
        # 取置信度最高的检测
        current = detections.pop(0)
        keep.append(current)

        # 移除重叠的检测（更严格的重叠判断）
        detections = [
            det for det in detections
            if calculate_iou(current['bbox'], det['bbox']) < iou_threshold
        ]

    return keep


def calculate_iou(box1, box2):
    """计算IoU"""
    # box格式: {x, y, width, height} - 已经是左上角坐标
    box1_x1 = box1['x']
    box1_y1 = box1['y']
    box1_x2 = box1['x'] + box1['width']
    box1_y2 = box1['y'] + box1['height']

    box2_x1 = box2['x']
    box2_y1 = box2['y']
    box2_x2 = box2['x'] + box2['width']
    box2_y2 = box2['y'] + box2['height']

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
    app.run(host='0.0.0.0', port=port, debug=False)