# app.py
from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
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


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


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
    """检测接口 - 纯ONNX Runtime"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # 检查文件
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 使用PIL读取图片
        try:
            image = Image.open(filepath).convert('RGB')
            original_size = image.size  # (width, height)
        except Exception as e:
            os.remove(filepath)
            return jsonify({'error': f'Invalid image: {str(e)}'}), 400

        # 预处理
        input_blob = preprocess(image)

        # 推理
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: input_blob})

        # 后处理
        boxes, scores, class_ids = postprocess(outputs)

        # 缩放回原始尺寸
        boxes = scale_boxes(boxes, original_size)

        # 构建响应
        results = []
        for i, box in enumerate(boxes):
            class_id = int(class_ids[i])
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"unknown_{class_id}"

            results.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': float(scores[i]),
                'bbox': {
                    'x1': float(box[0]),
                    'y1': float(box[1]),
                    'x2': float(box[2]),
                    'y2': float(box[3])
                }
            })

        # 清理文件
        os.remove(filepath)

        return jsonify({
            'success': True,
            'detections': results,
            'detection_count': len(results)
        })

    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
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
        'endpoint': '/detect (POST)'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)