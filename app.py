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
    session = ort.InferenceSession("best_compressed.onnx", providers=['CPUExecutionProvider'])
    print("✅ 压缩模型加载成功! (11.0MB)")
    input_name = session.get_inputs()[0].name
    print(f"模型输入: {input_name}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    session = None

class_names = ['paper', 'cup', 'citrus', 'bottle', 'battery']
class_names_chinese = ['纸张', '杯子', '果皮', '瓶子', '电池']


@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        print("📨 收到检测请求")

        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': '没有收到图片数据'})

        if session is None:
            return jsonify({'success': False, 'error': '模型未加载'})

        # 解码base64图片
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_data = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_data))
        original_size = image.size
        print(f"图片尺寸: {original_size}")

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_np = np.array(image)

        # 🔧 关键修复：使用与YOLOv8训练一致的预处理
        print("⚙️ 开始预处理...")
        input_tensor = preprocess_yolov8_compatible(image_np)

        # 模型推理
        print("🔮 开始模型推理...")
        outputs = session.run(None, {input_name: input_tensor})

        # 后处理 - 降低阈值
        print("📊 开始后处理...")
        detections = postprocess_yolov8(outputs[0], original_size, conf_threshold=0.15)  # 进一步降低阈值

        print(f"✅ 检测完成: {len(detections)} 个对象")

        return jsonify({
            'success': True,
            'detections': detections,
            'detection_count': len(detections),
            'classes_available': class_names_chinese
        })

    except Exception as e:
        print(f"❌ 检测错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


def preprocess_yolov8_compatible(image):
    """与YOLOv8训练完全一致的预处理"""
    # YOLOv8 标准预处理流程：
    # 1. BGR颜色通道（OpenCV默认）
    # 2. 直接调整大小到640x640（不保持比例）
    # 3. 归一化到0-1
    # 4. 通道顺序: HWC to CHW

    # 转换为BGR（YOLOv8训练时通常用OpenCV读取，是BGR格式）
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image

    # 调整大小到640x640
    resized = cv2.resize(image_bgr, (640, 640))

    # 归一化到0-1
    normalized = resized.astype(np.float32) / 255.0

    # 通道顺序: HWC to CHW
    chw = normalized.transpose(2, 0, 1)

    # 添加batch维度
    batch = np.expand_dims(chw, 0)

    print(f"🔧 预处理调试:")
    print(f"  - 输入范围: [{batch.min():.6f}, {batch.max():.6f}]")
    print(f"  - 输入形状: {batch.shape}")

    return batch


def postprocess_yolov8(outputs, original_size, conf_threshold=0.15, iou_threshold=0.45):
    """YOLOv8后处理"""
    try:
        print(f"🔍 模型输出形状: {outputs.shape}")

        # 输出是 [1, 9, 8400]，转置为 [8400, 9]
        outputs = outputs[0].transpose(1, 0)
        print(f"🔍 转置后形状: {outputs.shape}")
        print(f"🔍 输出范围: [{outputs.min():.6f}, {outputs.max():.6f}]")

        detections = []
        original_w, original_h = original_size

        # 计算尺度因子
        scale_x = original_w / 640
        scale_y = original_h / 640

        # 详细调试信息
        print("🔍 检查前20个检测:")
        valid_low_conf = 0
        for i in range(min(20, outputs.shape[0])):
            detection = outputs[i]
            bbox = detection[:4]
            obj_conf = detection[4]
            class_probs = detection[5:]

            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id]
            total_confidence = obj_conf * class_conf

            if total_confidence > 0.01:  # 非常低的阈值查看所有信号
                valid_low_conf += 1
                cx, cy, w, h = bbox
                print(
                    f"  检测{i}: 置信度={total_confidence:.4f}, 类别={class_id}({class_names[class_id]}), bbox=({cx:.1f},{cy:.1f},{w:.1f},{h:.1f})")

        print(f"🔍 极低阈值(0.01)下候选数: {valid_low_conf}")

        # 正式处理
        for i in range(outputs.shape[0]):
            detection = outputs[i]
            bbox = detection[:4]
            obj_conf = detection[4]
            class_probs = detection[5:]

            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id]
            total_confidence = obj_conf * class_conf

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

                if bbox_width >= 3 and bbox_height >= 3 and class_id < len(class_names):
                    detections.append({
                        'class': int(class_id),
                        'name': class_names[class_id],
                        'chineseName': class_names_chinese[class_id],
                        'confidence': round(float(total_confidence), 4),
                        'bbox': {
                            'x': round(float(x1), 1),
                            'y': round(float(y1), 1),
                            'width': round(float(bbox_width), 1),
                            'height': round(float(bbox_height), 1)
                        }
                    })

        print(f"🔍 阈值({conf_threshold})下检测数: {len(detections)}")

        # NMS
        if detections:
            detections = non_max_suppression_fast(detections, iou_threshold)
            detections.sort(key=lambda x: x['confidence'], reverse=True)

        return detections

    except Exception as e:
        print(f"后处理错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def non_max_suppression_fast(detections, iou_threshold):
    """NMS"""
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

        iou = intersection / (union + 1e-6)
        remaining_indices = np.where(iou <= iou_threshold)[0]
        indices = indices[remaining_indices + 1]

    return [detections[i] for i in keep]


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"🚀 启动服务在端口 {port}")
    app.run(host='0.0.0.0', port=port, debug=False)