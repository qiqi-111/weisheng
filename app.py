# app.py - 最终生产版本
import onnxruntime as ort
from flask import Flask, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 配置
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# 创建上传文件夹
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 模型配置
MODEL_PATH = 'best_compressed.onnx'
CLASS_NAMES = ['paper', 'cup', 'citrus', 'bottle', 'battery']

# NMS参数 - 使用经过验证的通用值
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


# 加载模型
def load_model():
    try:
        logger.info("正在加载模型...")
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        logger.info("✅ 模型加载成功")
        return session
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        raise


model = load_model()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess(image, input_size=(640, 640)):
    """图像预处理"""
    resized = cv2.resize(image, input_size).astype(np.float32) / 255.0
    blob = resized.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)
    return blob


def postprocess(outputs, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD):
    """后处理函数"""
    try:
        output = outputs[0]  # [1, 9, 8400]
        predictions = np.squeeze(output, 0).T  # [8400, 9]

        # 提取预测结果
        boxes = predictions[:, 0:4]
        scores = predictions[:, 4:]

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

        # 坐标转换
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        boxes_corner = np.column_stack([x1, y1, x2, y2])

        # 应用NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_corner.tolist(),
            confidences.tolist(),
            conf_threshold,
            iou_threshold
        )

        if len(indices) > 0:
            indices = indices.flatten()
            return boxes_corner[indices], confidences[indices], class_ids[indices]
        else:
            return [], [], []

    except Exception as e:
        logger.error(f"后处理错误: {e}")
        return [], [], []


def scale_boxes(boxes, original_shape, input_size=640):
    """缩放检测框"""
    if len(boxes) == 0:
        return boxes

    orig_height, orig_width = original_shape[:2]
    scale = min(input_size / orig_width, input_size / orig_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    dx = (input_size - new_width) / 2
    dy = (input_size - new_height) / 2

    scaled_boxes = boxes.copy()
    scaled_boxes[:, [0, 2]] = (scaled_boxes[:, [0, 2]] - dx) / scale
    scaled_boxes[:, [1, 3]] = (scaled_boxes[:, [1, 3]] - dy) / scale

    scaled_boxes[:, [0, 2]] = np.clip(scaled_boxes[:, [0, 2]], 0, orig_width)
    scaled_boxes[:, [1, 3]] = np.clip(scaled_boxes[:, [1, 3]], 0, orig_height)

    return scaled_boxes


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': True})


@app.route('/detect', methods=['POST'])
def detect_objects():
    """垃圾检测API - 完全适配小程序"""
    try:
        logger.info("收到检测请求")

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 读取图片
        original_image = cv2.imread(filepath)
        if original_image is None:
            os.remove(filepath)
            return jsonify({'error': 'Invalid image file'}), 400

        original_shape = original_image.shape

        # 预处理和推理
        input_blob = preprocess(original_image)
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: input_blob})

        # 后处理
        boxes, scores, class_ids = postprocess(outputs)
        boxes = scale_boxes(boxes, original_shape)

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
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'Garbage Detection API',
        'status': 'running',
        'endpoint': '/detect'
    })


if __name__ == '__main__':
    logger.info("启动服务...")
    app.run(host='0.0.0.0', port=8000, debug=False)