import cv2
import numpy as np
import onnxruntime as ort
import base64
from PIL import Image
import io
import json
import traceback

# 加载模型
print("正在加载YOLO模型...")
try:
    session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
    print("YOLO模型加载成功!")
except Exception as e:
    print(f"模型加载失败: {e}")
    session = None

# 类别名称
class_names = ['paper', 'cup', 'citrus', 'bottle', 'battery']


def diagnose_model():
    """诊断模型质量"""
    if session is None:
        print("模型未加载")
        return

    print("\n" + "=" * 50)
    print("模型诊断报告")
    print("=" * 50)

    # 1. 检查模型基本信息
    print("\n1. 模型基本信息:")
    print(f"输入数量: {len(session.get_inputs())}")
    for i, input_info in enumerate(session.get_inputs()):
        print(f"  输入[{i}]: 名称={input_info.name}, 形状={input_info.shape}, 类型={input_info.type}")

    print(f"输出数量: {len(session.get_outputs())}")
    for i, output_info in enumerate(session.get_outputs()):
        print(f"  输出[{i}]: 名称={output_info.name}, 形状={output_info.shape}, 类型={output_info.type}")

    # 2. 创建测试图像
    print("\n2. 创建测试图像...")
    # 创建纯色测试图像
    test_images = {
        '纯黑背景': np.zeros((640, 640, 3), dtype=np.uint8),
        '纯白背景': np.ones((640, 640, 3), dtype=np.uint8) * 255,
        '灰色背景': np.ones((640, 640, 3), dtype=np.uint8) * 128,
        '随机噪声': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    }

    for name, test_image in test_images.items():
        print(f"\n测试图像: {name}")
        analyze_model_output(test_image, name)

    # 3. 分析输出分布
    print("\n3. 输出分布分析:")
    analyze_output_distribution()


def analyze_model_output(image, image_name):
    """分析模型在特定图像上的输出"""
    try:
        # 预处理
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_float = image_rgb.astype(np.float32) / 255.0
        image_chw = image_float.transpose(2, 0, 1)
        input_tensor = np.expand_dims(image_chw, 0)

        # 推理
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
        output_data = outputs[0]  # [1, 9, 8400]

        # 分析输出
        output_flat = output_data.flatten()

        print(f"  {image_name}输出分析:")
        print(f"    数值范围: {output_flat.min():.6f} ~ {output_flat.max():.6f}")
        print(f"    平均值: {output_flat.mean():.6f}")
        print(f"    标准差: {output_flat.std():.6f}")

        # 分析类别分数部分
        output_reshaped = output_data[0].transpose(1, 0)  # [8400, 9]
        class_scores = output_reshaped[:, 4:]  # 只取类别分数部分

        # 应用sigmoid
        class_scores_sigmoid = 1.0 / (1.0 + np.exp(-class_scores))

        print(f"    类别分数范围(sigmoid前): {class_scores.min():.6f} ~ {class_scores.max():.6f}")
        print(f"    类别分数范围(sigmoid后): {class_scores_sigmoid.min():.6f} ~ {class_scores_sigmoid.max():.6f}")

        # 统计高置信度检测数量
        high_conf_count = np.sum(class_scores_sigmoid > 0.5)
        medium_conf_count = np.sum(class_scores_sigmoid > 0.25)
        low_conf_count = np.sum(class_scores_sigmoid > 0.1)

        print(f"    高置信度(>0.5)检测: {high_conf_count}")
        print(f"    中置信度(>0.25)检测: {medium_conf_count}")
        print(f"    低置信度(>0.1)检测: {low_conf_count}")

        # 检查是否有明显的误检模式
        if high_conf_count > 100:
            print(f"    ⚠️ 警告: 在{image_name}上检测到大量高置信度误检!")
        elif high_conf_count > 10:
            print(f"    ⚠️ 注意: 在{image_name}上检测到较多高置信度误检")
        else:
            print(f"    ✅ 在{image_name}上误检较少")

    except Exception as e:
        print(f"分析{image_name}时出错: {e}")


def analyze_output_distribution():
    """分析输出分布特征"""
    try:
        # 创建中性测试图像
        test_image = np.ones((640, 640, 3), dtype=np.uint8) * 128
        image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        image_float = image_rgb.astype(np.float32) / 255.0
        image_chw = image_float.transpose(2, 0, 1)
        input_tensor = np.expand_dims(image_chw, 0)

        # 推理
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
        output_data = outputs[0][0]  # [9, 8400]

        print("\n4. 详细输出分布:")

        # 分析每个特征维度的分布
        features = ['中心x', '中心y', '宽度w', '高度h', 'class0', 'class1', 'class2', 'class3', 'class4']

        for i, feature_name in enumerate(features):
            feature_data = output_data[i]
            print(f"  {feature_name}:")
            print(f"    范围: {feature_data.min():.3f} ~ {feature_data.max():.3f}")
            print(f"    均值: {feature_data.mean():.3f}, 标准差: {feature_data.std():.3f}")

            # 对于类别分数，分析sigmoid后的分布
            if i >= 4:
                sigmoid_data = 1.0 / (1.0 + np.exp(-feature_data))
                high_conf = np.sum(sigmoid_data > 0.7)
                print(f"    sigmoid后>0.7: {high_conf}")

        # 分析边界框尺寸分布
        print(f"\n5. 边界框尺寸分析:")
        widths = output_data[2]  # 宽度
        heights = output_data[3]  # 高度

        # 转换为像素尺寸（假设640x640）
        widths_px = widths * 640
        heights_px = heights * 640

        print(f"  宽度范围: {widths_px.min():.1f} ~ {widths_px.max():.1f} 像素")
        print(f"  高度范围: {heights_px.min():.1f} ~ {heights_px.max():.1f} 像素")

        # 统计异常尺寸
        too_small = np.sum((widths_px < 10) | (heights_px < 10))
        too_large = np.sum((widths_px > 600) | (heights_px > 600))
        reasonable = np.sum((widths_px >= 10) & (widths_px <= 600) &
                            (heights_px >= 10) & (heights_px <= 600))

        print(f"  过小框(<10px): {too_small}")
        print(f"  过大框(>600px): {too_large}")
        print(f"  合理框: {reasonable}")

    except Exception as e:
        print(f"分析输出分布时出错: {e}")


def test_with_real_image():
    """用真实图像测试"""
    print("\n6. 真实图像测试:")
    try:
        # 创建一个简单的测试图像 - 在灰色背景上画一些几何形状
        test_image = np.ones((640, 640, 3), dtype=np.uint8) * 128

        # 画一些形状模拟垃圾
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 0, 0), -1)  # 蓝色矩形
        cv2.circle(test_image, (400, 300), 50, (0, 255, 0), -1)  # 绿色圆形
        cv2.rectangle(test_image, (300, 400), (350, 450), (0, 0, 255), -1)  # 红色矩形

        # 分析这个图像
        analyze_model_output(test_image, "模拟垃圾图像")

    except Exception as e:
        print(f"真实图像测试出错: {e}")


if __name__ == '__main__':
    diagnose_model()
    test_with_real_image()

    print("\n" + "=" * 50)
    print("诊断总结:")
    print("=" * 50)
    print("""
可能的模型问题:
1. 训练数据不足 - 模型没有学到有意义的特征
2. 训练数据质量差 - 标注不准确或噪声太多  
3. 训练过程有问题 - 学习率、epoch等参数设置不当
4. 类别不平衡 - 某些类别样本太少
5. 过拟合 - 在训练集上表现好但泛化能力差

建议解决方案:
1. 检查训练数据质量和数量
2. 增加数据增强
3. 调整训练参数
4. 使用预训练权重
5. 增加正则化防止过拟合
    """)