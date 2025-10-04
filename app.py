from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import onnxruntime as ort
import base64
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

print("正在加载YOLO模型...")
try:
    # 使用压缩后的模型
    session = ort.InferenceSession("best_compressed.onnx", providers=['CPUExecutionProvider'])
    print("✅ 压缩模型加载成功! (11.0MB)")
    # 打印输入输出信息
    for i, input_info in enumerate(session.get_inputs()):
        print(f"输入 {i}: {input_info.name} 形状: {input_info.shape}")
    for i, output_info in enumerate(session.get_outputs()):
        print(f"输出 {i}: {output_info.name} 形状: {output_info.shape}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    session = None

# 类别名称（根据你的训练数据）
class_names = ['paper', 'cup', 'citrus', 'bottle', 'battery']
class_names_chinese = ['纸张', '杯子', '果皮', '瓶子', '电池']


@app.route('/')
def home():
    return jsonify({
        'message': '垃圾检测API服务运行中',
        'status': 'active',
        'model_loaded': session is not None,
        'model_size': '11.0MB (压缩版)',
        'classes': class_names_chinese
    })


@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'GET':
        return jsonify({'message': '请使用POST方法上传图片'})

    try:
        # 接收JSON数据
        data = request.get_json()
        print("📨 收到检测请求")

        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': '没有收到图片数据'})

        if session is None:
            return jsonify({'success': False, 'error': '模型未加载'})

        # 解码base64图片
        print("🖼️ 开始解码图片...")
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_data = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_data))
        original_size = image.size
        print(f"图片尺寸: {original_size}")

        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_np = np.array(image)

        # 预处理
        print("⚙️ 开始预处理...")
        input_tensor = preprocess(image_np)

        # 模型推理
        print("🔮 开始模型推理...")
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})

        # 后处理 - 关键修复！
        print("📊 开始后处理...")
        detections = postprocess_yolov8(outputs[0], original_size)
        print(f"✅ 检测完成: {len(detections)} 个对象")

        return jsonify({
            'success': True,
            'detections': detections,
            'model_info': 'yolov8_compressed_onnx',
            'detection_count': len(detections),
            'classes_available': class_names_chinese
        })

    except Exception as e:
        print(f"❌ 检测错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })


def preprocess(image):
    """YOLO预处理"""
    # 调整大小到640x640
    image = cv2.resize(image, (640, 640))
    # 归一化
    image = image.astype(np.float32) / 255.0
    # HWC to CHW
    image = image.transpose(2, 0, 1)
    # 添加batch维度
    image = np.expand_dims(image, 0)
    return image


def postprocess_yolov8(outputs, original_size, conf_threshold=0.25, iou_threshold=0.45):
    """YOLOv8专用后处理 - 修复版本"""
    try:
        # YOLOv8输出格式: [1, 84, 8400]
        # 84 = 4(bbox) + 80(classes)，但你只有5个类别，所以是9
        print(f"🔍 模型输出形状: {outputs.shape}")

        # 输出是 [1, 9, 8400]，我们需要转置为 [8400, 9]
        outputs = outputs[0].transpose(1, 0)  # [8400, 9]
        print(f"🔍 转置后形状: {outputs.shape}")

        detections = []
        original_width, original_height = original_size

        # 计算尺度因子
        scale_x = original_width / 640
        scale_y = original_height / 640

        # 调试：打印前几个检测的置信度
        print("🔍 前5个检测的置信度:")
        for i in range(min(5, outputs.shape[0])):
            detection = outputs[i]
            obj_conf = detection[4]
            class_probs = detection[5:]
            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id]
            total_confidence = obj_conf * class_conf
            print(f"  检测{i}: 总置信度={total_confidence:.3f}, 类别={class_id}({class_names[class_id]})")

        for i in range(outputs.shape[0]):
            detection = outputs[i]

            # 提取边界框坐标 (cx, cy, w, h)
            bbox = detection[:4]

            # 提取对象置信度
            obj_conf = detection[4]

            # 提取类别概率
            class_probs = detection[5:]

            # 找到最可能的类别
            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id]

            # 计算总置信度
            total_confidence = obj_conf * class_conf

            # 应用置信度阈值 - 降低到0.25提高灵敏度
            if total_confidence > conf_threshold:
                cx, cy, w, h = bbox

                # 转换为绝对像素坐标
                x_center = cx * scale_x
                y_center = cy * scale_y
                bbox_width = w * scale_x
                bbox_height = h * scale_y

                # 转换为左上角坐标
                x1 = max(0, x_center - bbox_width / 2)
                y1 = max(0, y_center - bbox_height / 2)

                # 确保边界框在图像范围内
                x1 = min(x1, original_width)
                y1 = min(y1, original_height)
                bbox_width = min(bbox_width, original_width - x1)
                bbox_height = min(bbox_height, original_height - y1)

                # 检查边界框是否合理
                if bbox_width >= 5 and bbox_height >= 5 and class_id < len(class_names):
                    detections.append({
                        'class': int(class_id),
                        'name': class_names[class_id],
                        'chineseName': class_names_chinese[class_id],
                        'confidence': round(float(total_confidence), 3),
                        'bbox': {
                            'x': round(float(x1), 1),
                            'y': round(float(y1), 1),
                            'width': round(float(bbox_width), 1),
                            'height': round(float(bbox_height), 1)
                        }
                    })

        print(f"🔍 应用阈值前检测数: {len(detections)}")

        # 应用NMS
        if detections:
            detections = non_max_suppression_fast(detections, iou_threshold)
            print(f"🔍 应用NMS后检测数: {len(detections)}")

            # 按置信度排序
            detections.sort(key=lambda x: x['confidence'], reverse=True)

            # 限制返回数量
            if len(detections) > 20:
                detections = detections[:20]

        return detections

    except Exception as e:
        print(f"后处理错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def non_max_suppression_fast(detections, iou_threshold):
    """快速非极大值抑制"""
    if len(detections) == 0:
        return []

    boxes = np.array([[det['bbox']['x'], det['bbox']['y'],
                       det['bbox']['x'] + det['bbox']['width'],
                       det['bbox']['y'] + det['bbox']['height']] for det in detections])
    scores = np.array([det['confidence'] for det in detections])

    indices = np.argsort(scores)[::-1]
    keep = []

    while indices.size > 0:
        current = indices[0]
        keep.append(current)

        if indices.size == 1:
            break

        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]

        xx1 = np.maximum(current_box[0], other_boxes[:, 0])
        yy1 = np.maximum(current_box[1], other_boxes[:, 1])
        xx2 = np.minimum(current_box[2], other_boxes[:, 2])
        yy2 = np.minimum(current_box[3], other_boxes[:, 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        union = current_area + other_areas - intersection

        iou = intersection / (union + 1e-6)  # 避免除零

        remaining_indices = np.where(iou <= iou_threshold)[0]
        indices = indices[remaining_indices + 1]

    return [detections[i] for i in keep]


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': session is not None,
        'model_type': 'compressed_onnx',
        'classes': class_names_chinese
    })


@app.route('/debug_detect', methods=['POST'])
def debug_detect():
    """调试接口，返回更详细的信息"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': '没有收到图片数据'})

        if session is None:
            return jsonify({'success': False, 'error': '模型未加载'})

        # 解码图片
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_data = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_data))
        original_size = image.size

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_np = np.array(image)
        input_tensor = preprocess(image_np)

        # 模型推理
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})

        # 原始输出信息
        raw_output = outputs[0]
        print(f"🔧 调试信息 - 原始输出形状: {raw_output.shape}")
        print(f"🔧 调试信息 - 原始输出范围: [{raw_output.min():.3f}, {raw_output.max():.3f}]")

        # 测试不同阈值
        results = {}
        for conf_thresh in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
            detections = postprocess_yolov8(outputs[0], original_size, conf_thresh, 0.45)
            results[f'conf_{conf_thresh}'] = len(detections)

        return jsonify({
            'success': True,
            'original_size': original_size,
            'output_shape': str(raw_output.shape),
            'output_range': [float(raw_output.min()), float(raw_output.max())],
            'detections_at_different_thresholds': results,
            'message': '调试信息已输出到控制台'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"🚀 启动服务在端口 {port}")
    print(f"📋 可用类别: {class_names_chinese}")
    app.run(host='0.0.0.0', port=port, debug=False)