# app.py
from flask import Flask, request, jsonify
import subprocess
import os
import json
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


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def parse_yolo_output(image_path):
    """解析YOLO命令的输出"""
    try:
        # 运行YOLO检测命令
        cmd = f"yolo predict model={MODEL_PATH} source='{image_path}' save=False"
        print(f"执行命令: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

        print(f"返回码: {result.returncode}")
        print(f"标准输出: {result.stdout}")
        if result.stderr:
            print(f"错误输出: {result.stderr}")

        if result.returncode != 0:
            return []

        # 解析YOLO输出 - 根据实际输出格式调整
        output = result.stdout
        detections = []

        # 方法1: 查找包含类别名的行
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            # 查找包含类别名称的行
            for class_name in CLASS_NAMES:
                if class_name in line.lower():
                    # 尝试解析置信度和坐标
                    parts = line.split()
                    confidence = None
                    bbox = []

                    # 查找置信度 (通常以%或小数形式)
                    for part in parts:
                        if '%' in part:
                            confidence = float(part.replace('%', '')) / 100.0
                            break
                        elif part.replace('.', '').isdigit() and 0 <= float(part) <= 1:
                            confidence = float(part)

                    # 查找坐标数字
                    for part in parts:
                        if part.replace('.', '').replace('-', '').isdigit():
                            bbox.append(float(part))
                            if len(bbox) >= 4:
                                break

                    if confidence is not None and len(bbox) >= 4:
                        class_id = CLASS_NAMES.index(class_name)
                        detections.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': {
                                'x1': bbox[0],
                                'y1': bbox[1],
                                'x2': bbox[2],
                                'y2': bbox[3]
                            }
                        })
                    break

        # 方法2: 如果方法1没找到，尝试JSON格式解析
        if not detections and '[' in output and ']' in output:
            try:
                # 尝试从输出中提取JSON数据
                start = output.find('[')
                end = output.find(']') + 1
                json_str = output[start:end]
                json_data = json.loads(json_str)

                for item in json_data:
                    if isinstance(item, dict) and 'class' in item:
                        class_name = item.get('class', '')
                        if class_name in CLASS_NAMES:
                            class_id = CLASS_NAMES.index(class_name)
                            detections.append({
                                'class_id': class_id,
                                'class_name': class_name,
                                'confidence': item.get('confidence', 0.5),
                                'bbox': item.get('bbox', {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0})
                            })
            except:
                pass

        print(f"解析到 {len(detections)} 个检测结果")
        return detections

    except subprocess.TimeoutExpired:
        print("YOLO命令执行超时")
        return []
    except Exception as e:
        print(f"解析YOLO输出错误: {e}")
        return []


@app.route('/detect', methods=['POST'])
def detect_objects():
    """检测接口 - 使用YOLO命令"""
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
        print(f"文件保存到: {filepath}")

        # 使用YOLO命令进行检测
        detections = parse_yolo_output(filepath)

        # 清理文件
        os.remove(filepath)

        return jsonify({
            'success': True,
            'detections': detections,
            'detection_count': len(detections)
        })

    except Exception as e:
        print(f"服务器错误: {e}")
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Service is running'})


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'Garbage Detection API',
        'status': 'running',
        'endpoint': '/detect (POST)'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  # 修正为8080端口