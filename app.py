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
    # 修复：添加 providers 参数
    session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
    print("YOLO模型加载成功!")

    # 打印模型输入输出信息
    print("模型输入信息:")
    for input_info in session.get_inputs():
        print(f"  名称: {input_info.name}, 形状: {input_info.shape}, 类型: {input_info.type}")

    print("模型输出信息:")
    for output_info in session.get_outputs():
        print(f"  名称: {output_info.name}, 形状: {output_info.shape}, 类型: {output_info.type}")

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
        detections = postprocess(outputs)
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
        # 已经是RGB，不需要转换
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
    """通用YOLO后处理 - 尝试多种输出格式"""
    try:
        print("=== 开始后处理 ===")

        # 检查所有输出
        for i, output in enumerate(outputs):
            print(f"输出[{i}] 形状: {output.shape}, 数据类型: {output.dtype}")
            print(f"输出[{i}] 数值范围: {output.min():.6f} ~ {output.max():.6f}")

            # 检查前几个值的样本
            if len(output.shape) == 3:
                sample = output[0, :, :5]  # 取前5个检测点的前5个值
                print(f"输出[{i}] 样本数据 (前5个检测点):")
                for j in range(min(5, sample.shape[0])):
                    print(f"  检测点 {j}: {sample[j]}")

        # 尝试不同的输出格式解析
        detections = []

        # 方法1: 尝试YOLOv8格式 [1, 9, 8400]
        if len(outputs) == 1 and outputs[0].shape == (1, 9, 8400):
            print("尝试YOLOv8格式解析...")
            detections = parse_yolov8_output(outputs[0], conf_threshold)

        # 方法2: 尝试YOLOv5格式 [1, 25200, 85]
        elif len(outputs) == 1 and outputs[0].shape == (1, 25200, 85):
            print("尝试YOLOv5格式解析...")
            detections = parse_yolov5_output(outputs[0], conf_threshold)

        # 方法3: 尝试其他可能的格式
        else:
            print("尝试通用格式解析...")
            for i, output in enumerate(outputs):
                if len(output.shape) == 3:
                    detections.extend(parse_generic_output(output, conf_threshold, f"output_{i}"))

        print(f"初步检测到 {len(detections)} 个对象")

        # 应用NMS
        if detections:
            result = non_max_suppression(detections)
            print(f"NMS后剩余 {len(result)} 个对象")
            return result
        else:
            print("未检测到任何对象")
            return []

    except Exception as e:
        print(f"后处理错误: {str(e)}")
        print(f"错误堆栈: {traceback.format_exc()}")
        return []


def parse_yolov8_output(output, conf_threshold):
    """解析YOLOv8输出格式 [1, 9, 8400]"""
    detections = []
    output_data = output[0]  # [9, 8400]
    output_data = output_data.transpose(1, 0)  # [8400, 9]

    print(f"YOLOv8解析: 转换后形状 {output_data.shape}")

    for i in range(output_data.shape[0]):
        detection = output_data[i]
        confidence = detection[4]

        if confidence > conf_threshold:
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            final_confidence = confidence * class_confidence

            if final_confidence > conf_threshold and class_id < len(class_names):
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

    return detections


def parse_yolov5_output(output, conf_threshold):
    """解析YOLOv5输出格式 [1, 25200, 85]"""
    detections = []
    output_data = output[0]  # [25200, 85]

    print(f"YOLOv5解析: 输出形状 {output_data.shape}")

    for i in range(output_data.shape[0]):
        detection = output_data[i]
        confidence = detection[4]

        if confidence > conf_threshold:
            class_scores = detection[5:5 + len(class_names)]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            final_confidence = confidence * class_confidence

            if final_confidence > conf_threshold and class_id < len(class_names):
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

    return detections


def parse_generic_output(output, conf_threshold, output_name):
    """通用输出解析"""
    detections = []

    if len(output.shape) == 3:  # [batch, num_detections, features]
        output_data = output[0]  # 去掉batch维度

        print(f"通用解析 {output_name}: 形状 {output_data.shape}")

        # 尝试推断特征维度
        num_features = output_data.shape[1]
        print(f"特征维度: {num_features}")

        # 假设最后几个维度是类别分数
        if num_features >= 5 + len(class_names):
            for i in range(output_data.shape[0]):
                detection = output_data[i]
                confidence = detection[4]

                if confidence > conf_threshold:
                    class_scores = detection[5:5 + len(class_names)]
                    class_id = np.argmax(class_scores)
                    class_confidence = class_scores[class_id]
                    final_confidence = confidence * class_confidence

                    if final_confidence > conf_threshold and class_id < len(class_names):
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

    return detections


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