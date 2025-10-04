# app.py
from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import base64
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 基础配置
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# 创建目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 模型路径
MODEL_PATH = 'best_compressed.onnx'
CLASS_NAMES = ['paper', 'cup', 'citrus', 'bottle', 'battery']

# 加载模型
try:
    model = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    model = None


def base64_to_image(base64_string):
    """将Base64字符串转换为PIL图像"""
    try:
        # 移除可能的数据URL前缀
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        # 解码Base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return image
    except Exception as e:
        print(f"Base64转换错误: {e}")
        return None


def preprocess(image):
    """使用PIL进行预处理"""
    # 调整大小到640x640
    image = image.resize((640, 640))
    # 转换为numpy数组
    img_array = np.array(image).astype(np.float32)
    # 归一化
    img_array /= 255.0
    # 转换维度 (H, W, C) -> (C, H, W)
    img_array = img_array.transpose(2, 0, 1)
    # 添加batch维度
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def nms(boxes, scores, iou_threshold=0.45):
    """简单的NMS实现"""
    if len(boxes) == 0:
        return []

    # 按置信度排序
    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # 计算IoU
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])

        iou = inter / (area_i + area_j - inter + 1e-6)

        # 保留IoU小于阈值的框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def postprocess(outputs, conf_threshold=0.25):
    """后处理函数"""
    if model is None:
        return [], [], []

    output = outputs[0]  # [1, 9, 8400]
    predictions = np.squeeze(output, 0).T  # [8400, 9]

    # 提取边界框和分数
    boxes = predictions[:, 0:4]  # [8400, 4] - cx, cy, w, h
    scores = predictions[:, 4:]  # [8400, 5] - 类别概率

    # 找到最佳类别和置信度
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # 过滤低置信度检测
    valid_mask = confidences > conf_threshold
    boxes = boxes[valid_mask]
    confidences = confidences[valid_mask]
    class_ids = class_ids[valid_mask]

    if len(boxes) == 0:
        return [], [], []

    # 将中心坐标转换为角坐标
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    boxes_corner = np.column_stack([x1, y1, x2, y2])

    # 应用NMS
    indices = nms(boxes_corner, confidences)

    if len(indices) > 0:
        return boxes_corner[indices], confidences[indices], class_ids[indices]
    else:
        return [], [], []


def scale_boxes(boxes, original_shape):
    """将检测框缩放回原始图像尺寸"""
    if len(boxes) == 0:
        return boxes

    orig_height, orig_width = original_shape
    input_size = 640

    # 计算缩放比例
    scale = min(input_size / orig_width, input_size / orig_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    # 计算填充
    dx = (input_size - new_width) / 2
    dy = (input_size - new_height) / 2

    # 缩放并调整填充
    scaled_boxes = boxes.copy()
    scaled_boxes[:, [0, 2]] = (scaled_boxes[:, [0, 2]] - dx) / scale
    scaled_boxes[:, [1, 3]] = (scaled_boxes[:, [1, 3]] - dy) / scale

    # 确保坐标在图像范围内
    scaled_boxes[:, [0, 2]] = np.clip(scaled_boxes[:, [0, 2]], 0, orig_width)
    scaled_boxes[:, [1, 3]] = np.clip(scaled_boxes[:, [1, 3]], 0, orig_height)

    return scaled_boxes


@app.route('/detect', methods=['POST'])
def detect_objects():
    """检测接口 - 支持Base64和文件上传"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        image = None
        original_size = None

        # 检查是否是Base64格式
        if request.content_type == 'application/json':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400

            # 从JSON中获取Base64图片
            base64_image = data.get('image') or data.get('base64') or data.get('file')
            if not base64_image:
                return jsonify({'error': 'No image data in JSON'}), 400

            print(f"收到Base64图片，长度: {len(base64_image)}")
            image = base64_to_image(base64_image)
            if image is None:
                return jsonify({'error': 'Invalid base64 image'}), 400

            original_size = image.size  # (width, height)

        # 检查是否是文件上传
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            # 保存临时文件
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 读取图片
            try:
                image = Image.open(filepath).convert('RGB')
                original_size = image.size
            except Exception as e:
                os.remove(filepath)
                return jsonify({'error': f'Invalid image: {str(e)}'}), 400

            # 清理临时文件
            os.remove(filepath)

        else:
            return jsonify({'error': 'No image data provided. Send JSON with base64 image or multipart file'}), 400

        if image is None:
            return jsonify({'error': 'Failed to process image'}), 400

        print(f"图片处理成功，尺寸: {original_size}")

        # 预处理
        input_blob = preprocess(image)

        # 推理
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: input_blob})
        print("推理完成")

        # 后处理
        boxes, scores, class_ids = postprocess(outputs)
        print(f"检测到 {len(boxes)} 个目标")

        # 缩放回原始尺寸
        boxes = scale_boxes(boxes, original_size)

        # 构建响应 - 适配你的小程序格式
        garbage_details = []
        for i, box in enumerate(boxes):
            class_id = int(class_ids[i])
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"unknown_{class_id}"

            # 转换为相对坐标 (0-1范围)
            width = original_size[0]
            height = original_size[1]

            garbage_details.append({
                'class': class_id,
                'name': class_name,
                'chineseName': class_name,  # 你可以在这里添加中文名称映射
                'confidence': float(scores[i]),
                'bbox': {
                    'x': float(box[0]) / width,
                    'y': float(box[1]) / height,
                    'width': float(box[2] - box[0]) / width,
                    'height': float(box[3] - box[1]) / height
                }
            })

        # 计算分数 (根据检测数量)
        garbage_count = len(garbage_details)
        score = max(0, 100 - garbage_count * 10)  # 每个垃圾扣10分

        return jsonify({
            'success': True,
            'score': score,
            'garbageCount': garbage_count,
            'garbageDetails': garbage_details,
            'modelUsed': 'onnx_model'
        })

    except Exception as e:
        print(f"服务器错误: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    status = 'healthy' if model is not None else 'model_not_loaded'
    return jsonify({'status': status, 'model_loaded': model is not None})


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'Garbage Detection API',
        'status': 'running' if model is not None else 'model_not_loaded',
        'endpoint': '/detect (POST)',
        'supported_formats': ['multipart/form-data', 'application/json (base64)']
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)