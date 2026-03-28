# ISP 版本对比模块详细设计

> **模块名称**: ISP Version Comparator
> **版本**: v1.0
> **日期**: 2026-03-28
> **状态**: 待石先生确认

---

## 一、模块定位

### 1.1 核心价值

| 价值 | 说明 |
|------|------|
| **自动化** | 减少 Imatest 人工操作，实现自动化批量对比 |
| **标准化** | 统一对比流程和评估标准 |
| **可追溯** | 记录每次对比结果，形成对比历史 |
| **智能辅助** | AI 分析结果，给出调参建议 |

### 1.2 与现有流程对比

| 维度 | 现有流程 (Imatest) | 目标流程 (ISP-Agent) |
|------|-------------------|-------------------|
| 操作方式 | 人工操作 GUI | 脚本自动化 |
| 对比维度 | 固定指标 | 可配置多维度 |
| 数据格式 | 需要手动导入 | 自动解析 comp12 |
| 输出形式 | 手动截图 | 自动生成报告 |
| 调参建议 | 无 | AI 智能建议 |
| 迭代跟踪 | 手动记录 | 自动历史追踪 |

---

## 二、数据格式规范

### 2.1 Comp12 RAW 格式

```
┌─────────────────────────────────────────────────────────────┐
│                    Comp12 RAW 数据格式                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Bit Depth: 12-bit                                          │
│  排列方式: 有效像素紧密排列，无填充位                         │
│  RGB Pattern:                                               │
│    - RGGB (最常见)                                          │
│    - BGGR                                                  │
│    - GRBG                                                  │
│    - GBRG                                                  │
│                                                              │
│  数据转换要求:                                              │
│    1. Comp12 → Raw16                                       │
│    2. 有效像素放置在低12位                                  │
│    3. 高4位补0                                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 CModel ISP 转换

```
Input: Comp12 RAW
       │
       ▼
┌─────────────────┐
│  Comp12 Parser  │
│  - 解析 Pattern │
│  - 12bit → 16bit│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Demosaic       │
│  - RGGB → RGB   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ISP Pipeline   │
│  (CModel)       │
└────────┬────────┘
         │
         ▼
   Output: RGB Image
```

### 2.3 支持的图像来源

| 来源 | 说明 | 用途 |
|------|------|------|
| **靶图** | 标准测试靶 (ISO 12233 等) | 精确指标测量 |
| **实景图** | 自然场景拍摄 | 主观评价参考 |
| **智驾场景** | 行车记录、对应场景 | ADAS 相关评估 |
| **座舱显示** | 车内屏幕显示 | HMI 相关评估 |

### 2.4 Golden Reference

| 类型 | 说明 | 使用方式 |
|------|------|----------|
| **标准靶图** | ISO 12233, Dot Chart, Color Chart | 精确测量 PSNR/SSIM |
| **主观基准** | 人工确认的"好"图像 | 作为对比基准 |
| **Previous Version** | 上一版 ISP 输出 | 版本差异对比 |

---

## 三、对比维度设计

### 3.1 核心对比维度

#### D1. 交通灯颜色还原

```
┌─────────────────────────────────────────────────────────────┐
│                    交通灯颜色还原评估                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  【场景1】ISP 参数调优                                       │
│    检测对象:                                                   │
│      - 红灯 (620-750nm)                                      │
│      - 黄灯 (580-595nm)                                      │
│      - 绿灯 (495-570nm)                                      │
│                                                              │
│    评估指标:                                                   │
│      1. 色度偏差 Δuv (CIE 1976)                              │
│      2. 亮度准确度                                            │
│      3. 颜色均匀性                                            │
│      4. 鬼影/光晕检测                                         │
│                                                              │
│  【场景2】ADAS 感知验证                                       │
│    输入: ISP 输出图像                                          │
│    处理: 送到感知模型进行交通灯检测                           │
│    输出:                                                      │
│      - 检测框位置                                            │
│      - 检测置信度                                            │
│      - 检测是否成功                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**两个场景的评估方式对比**:

| 维度 | 场景1: ISP 调优 | 场景2: ADAS 感知 |
|------|-----------------|------------------|
| 评估目标 | 颜色/轮廓是否正确 | 感知模型能否检出 |
| 评估方式 | 客观指标计算 | 感知模型输出 |
| 成功标准 | 指标达标 | 检测成功 |
| 典型用户 | ISP 工程师 | 算法工程师 |

#### D2. 轮廓评估

```
┌─────────────────────────────────────────────────────────────┐
│                    轮廓评估                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  评估内容:                                                   │
│    1. 边缘锐度 (Edge Sharpness)                             │
│       - MTF50 估计                                         │
│       - 10%-90% 上升时间                                   │
│                                                              │
│    2. 边缘完整性 (Edge Completeness)                        │
│       - 边缘连续性                                         │
│       - 边缘断裂检测                                        │
│                                                              │
│    3. 轮廓准确性 (Contour Accuracy)                        │
│       - 轮廓与实际形状的偏差                                │
│       - 几何失真估计                                        │
│                                                              │
│  输出:                                                       │
│    - 各边缘的 MTF 曲线                                      │
│    - 边缘锐度评分 (0-100)                                   │
│    - 异常边缘位置标注                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 感知模型集成

```
┌─────────────────────────────────────────────────────────────┐
│                    ADAS 感知模型集成                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                 ISP 输出图像                           │  │
│  └──────────────────────┬──────────────────────────────┘  │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              感知模型推理                            │  │
│  │  - YOLOv8 / YOLOX / CentreNet 等                  │  │
│  │  - 输入: RGB 图像                                   │  │
│  │  - 输出: 检测框 + 类别 + 置信度                     │  │
│  └──────────────────────┬──────────────────────────────┘  │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              交通灯专项分析                          │  │
│  │  - 红灯/黄灯/绿灯 检测结果                         │  │
│  │  - 检测置信度                                        │  │
│  │  - 检测框位置 (可选)                                │  │
│  │  - 与 GT 比较 (如果有)                              │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 感知模型接口设计

```python
class PerceptionModel:
    """感知模型接口"""
    
    def __init__(self, model_path, device="cuda"):
        """
        初始化感知模型
        
        参数:
            model_path: 模型文件路径 (.pt/.onnx)
            device: 推理设备 ("cuda" / "cpu")
        """
        pass
    
    def detect_traffic_lights(self, image):
        """
        检测交通灯
        
        参数:
            image: RGB 图像 (numpy array, HxWx3)
        
        返回:
            detections: [
                {
                    "bbox": [x1, y1, x2, y2],  # 检测框
                    "class": "red_light",        # 类别
                    "confidence": 0.95          # 置信度
                },
                ...
            ]
        """
        pass
    
    def evaluate_detection(self, detections, ground_truth=None):
        """
        评估检测结果
        
        返回:
            metrics: {
                "detected": true/false,
                "confidence": 0.95,
                "true_positive": true/false
            }
        """
        pass
```

#### 感知集成配置

```yaml
# 感知模型配置
perception:
  enabled: true
  model:
    type: "yolov8"
    path: "/models/traffic_light_yolov8.pt"
    device: "cuda"
    
  classes:
    - "traffic_light"
    - "red_light"
    - "yellow_light"  
    - "green_light"
  
  iou_threshold: 0.5
  confidence_threshold: 0.5
```

### 3.3 通用对比维度 (扩展)

| 维度 ID | 维度名称 | 说明 | 优先级 |
|---------|----------|------|--------|
| D3 | 清晰度 | 纹理细节保留 | P1 |
| D4 | 噪声控制 | 亮度/色度噪声 | P1 |
| D5 | 色彩准确 | 白平衡、饱和度 | P1 |
| D6 | 动态范围 | 高光/暗部细节 | P1 |
| D7 | 色彩均匀性 | 整个画面色彩一致性 | P2 |
| D8 | 几何畸变 | 枕形/桶形畸变 | P2 |
| D9 | 摩尔纹 | 周期性纹理伪影 | P2 |
| D10 | 紫边 | 色散/紫边检测 | P2 |

### 3.3 对比维度配置

```yaml
# 对比维度配置示例
comparison:
  enabled_dimensions:
    - traffic_light_color    # 交通灯颜色还原
    - contour               # 轮廓评估
    - sharpness             # 清晰度
    - noise                 # 噪声控制
    - color_accuracy        # 色彩准确
    - dynamic_range         # 动态范围
  
  thresholds:
    traffic_light_color:
      max_chroma_delta: 0.05  # 色度最大偏差
      min_brightness: 100      # 最小亮度
    contour:
      min_sharpness_score: 70  # 最小锐度评分
      max_edge_breakage: 5      # 最大边缘断裂数
```

---

## 四、功能流程设计

### 4.1 完整对比流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    ISP 版本对比完整流程                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐                                                │
│  │ Step 1      │  调试需求输入                                   │
│  │ 需求输入    │  - 场景描述                                     │
│  │             │  - 预期改善目标                                 │
│  │             │  - 参考图像 (可选)                            │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │ Step 2      │  ISP 参数调试                                   │
│  │ 参数调试    │  - AI 解读需求，制定调参策略                   │
│  │             │  - 输出建议的参数配置                          │
│  │             │  - 标注关键参数及影响                          │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │ Step 3      │  效果验证                                      │
│  │ 效果验证    │  - 调参后效果图生成                            │
│  │             │  - 与调参前对比                                │
│  │             │  - 是否达到预期改善                            │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐     ┌─────────────┐                           │
│  │ 达到预期    │────►│ Step 4      │                           │
│  │             │ yes │ Bench 对比   │                           │
│  └─────────────┘     │             │                           │
│         │            │ 与 Golden / │                           │
│         │ no         │ Previous Ver │                           │
│         ▼            └──────┬──────┘                           │
│  ┌─────────────┐            │                                    │
│  │ 继续迭代    │            ▼                                    │
│  │ 返回 Step 2 │     ┌─────────────┐                            │
│  └─────────────┘     │ Step 5      │                            │
│                      │ 人工确认    │                            │
│                      └──────┬──────┘                            │
│                             │                                    │
│         ┌──────────────────┼──────────────────┐               │
│         ▼                  ▼                  ▼               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │ 符合要求    │     │ 不符合要求  │     │ 差异过大    │        │
│  │             │     │             │     │             │        │
│  │ 记录调试经验 │     │ 返回 Step 2 │     │ 进一步分析  │        │
│  │ 生成报告    │     │ 继续迭代    │     │            │        │
│  └─────────────┘     └─────────────┘     └─────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 自动化对比流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    自动化对比流程 (可选)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  输入:                                                           │
│    - ISP_A 输出图像                                             │
│    - ISP_B 输出图像                                             │
│    - Golden Reference (可选)                                     │
│    - 配置: 对比维度、阈值                                       │
│                                                                  │
│  ┌─────────────┐                                                │
│  │ 数据准备    │                                                │
│  │ - 解析 RAW │                                                │
│  │ - 图像对齐 │                                                │
│  │ - ROI 选取 │                                                │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │ 指标计算    │                                                │
│  │ - 各维度指标│                                                │
│  │ - 差异热力图│                                                │
│  │ - 统计汇总  │                                                │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │ 报告生成   │                                                │
│  │ - 对比汇总表│                                                │
│  │ - 可视化   │                                                │
│  │ - AI 解读  │                                                │
│  └─────────────┘                                                │
│                                                                  │
│  输出:                                                           │
│    - 结构化对比报告 (JSON/Markdown)                             │
│    - 可视化对比图                                               │
│    - AI 分析建议                                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、评估指标详细设计

### 5.1 交通灯颜色还原

```python
class TrafficLightEvaluator:
    """交通灯颜色还原评估器"""
    
    # 标准颜色范围 (HSV 空间)
    RED_RANGE = {"lower": (0, 50, 50), "upper": (10, 255, 255)}
    YELLOW_RANGE = {"lower": (20, 50, 50), "upper": (30, 255, 255)}
    GREEN_RANGE = {"lower": (40, 50, 50), "upper": (80, 255, 255)}
    
    def evaluate(self, image, roi=None):
        """
        返回:
        {
            "red": {
                "detected": true,
                "center_hsv": (5, 200, 180),
                "chroma_delta": 0.023,    # 与标准色偏差
                "brightness": 175,
                "uniformity": 0.92,
                "passed": true
            },
            "yellow": { ... },
            "green": { ... },
            "overall_score": 85,
            "issues": []
        }
        """
        pass
```

### 5.2 轮廓评估

```python
class ContourEvaluator:
    """轮廓评估器"""
    
    def evaluate(self, image,roi=None):
        """
        返回:
        {
            "sharpness": {
                "mtf50_estimate": 0.45,    # MTF50 估计
                "rise_time_10_90": 2.3,    # 像素
                "score": 78
            },
            "completeness": {
                "edge_continuity": 0.95,  # 边缘连续性
                "breakage_count": 1,
                "breakage_locations": [(120, 340)],
                "score": 90
            },
            "accuracy": {
                "geometric_distortion": 0.02,  # 几何畸变量
                "contour_deviation_px": 1.2,
                "score": 85
            },
            "overall_score": 84,
            "issues": [
                {"type": "edge_breakage", "location": (120, 340), "severity": "low"}
            ]
        }
        """
        pass
```

### 5.3 通用指标

| 指标 | 方法 | 说明 |
|------|------|------|
| **PSNR** | `cv2.PSNR(img1, img2)` | 需要 Golden Reference |
| **SSIM** | `skimage.metrics.structural_similarity` | 结构相似性 |
| **BRISQUE** | `brisque.quality_IQA` | 无参考质量评分 |
| **LPIPS** | `lpips.LPIPS()` | 感知相似度 (深度学习) |
| **NIQE** | `niqe.compute_niqe` | 无参考自然图像质量 |

---

## 六、技术实现

### 6.1 Comp12 解析

```python
class Comp12Parser:
    """Comp12 RAW 格式解析器"""
    
    SUPPORTED_PATTERNS = ["RGGB", "BGGR", "GRBG", "GBRG"]
    
    def parse(self, filepath, width, height, pattern="RGGB"):
        """
        解析 Comp12 文件
        
        参数:
            filepath: Comp12 文件路径
            width: 图像宽度 (3840 或 1920)
            height: 图像高度 (2160 或 1080)
            pattern: Bayer 模式 (RGGB/BGGR/GRBG/GBRG)
        
        返回:
            raw16_image: 16-bit RAW 图像 (高4位为0)
        """
        # 1. 读取 12-bit 数据 (小端)
        raw12 = np.fromfile(filepath, dtype=np.uint16)
        
        # 2. 扩展到 16-bit (低12位，高4位补0)
        raw16 = raw12.astype(np.uint16)
        
        # 3. 重塑为 2D 图像
        raw16 = raw16.reshape((height, width))
        
        return raw16
    
    def save_for_cmodel(self, raw16_image, output_path):
        """保存为 CModel 可接受的格式"""
        raw16_image.tofile(output_path)
```

### 6.2 CModel ISP 调用

```python
class CModelISP:
    """CModel ISP 集成"""
    
    def __init__(self, cmodel_path, num_threads=None):
        """
        初始化 CModel ISP
        
        参数:
            cmodel_path: CModel 可执行文件路径
            num_threads: CPU 线程数 (默认: CPU 核心数)
        """
        self.cmodel_path = cmodel_path
        self.num_threads = num_threads or os.cpu_count()
    
    def process(self, raw_path, output_path, params=None):
        """
        调用 CModel 进行 ISP 处理
        
        参数:
            raw_path: 输入 RAW 文件路径
            output_path: 输出图像路径 (RGB 24bit)
            params: ISP 参数 (dict)
        
        返回:
            result: {"success": True, "output_path": output_path}
        """
        cmd = [
            self.cmodel_path,
            "-i", raw_path,
            "-o", output_path,
            "-threads", str(self.num_threads)
        ]
        
        if params:
            for key, value in params.items():
                cmd.extend([f"--{key}", str(value)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            raise RuntimeError(f"CModel failed: {result.stderr}")
        
        return {"success": True, "output_path": output_path}
    
    def batch_process(self, raw_files, output_dir, params=None):
        """
        批量处理 - 多线程并行
        
        参数:
            raw_files: RAW 文件路径列表
            output_dir: 输出目录
            params: ISP 参数
        
        返回:
            results: 处理结果列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self.process, raw, 
                    os.path.join(output_dir, f"output_{i}.jpg"), 
                    params): raw 
                for i, raw in enumerate(raw_files)
            }
            
            results = [f.result() for f in as_completed(futures)]
        
        return results
```

### 6.3 感知模型集成 (预留接口)

```python
class PerceptionModel:
    """感知模型接口 - 预留，最后接入"""
    
    def __init__(self, model_path=None, device="cpu"):
        """
        预留接口，最后接入感知模型
        
        参数:
            model_path: 模型文件路径 (.pt/.onnx)
            device: 推理设备 ("cuda" / "cpu")
        """
        self.model_path = model_path
        self.device = device
        self.enabled = model_path is not None
    
    def detect_traffic_lights(self, image):
        """
        检测交通灯
        
        返回:
            detections: [{"bbox": [x1,y1,x2,y2], "class": "red_light", "confidence": 0.95}]
        """
        if not self.enabled:
            return []  # 预留接口，未接入时返回空
        raise NotImplementedError("Perception model not yet integrated")
```

---

## 七、输出报告设计

### 7.1 对比报告结构

```json
{
  "report_id": "ISP_COMP_20260328_001",
  "timestamp": "2026-03-28T10:30:00+08:00",
  
  "comparison_type": "version_diff",
  "versions": {
    "current": "ISP_v2.1",
    "reference": "ISP_v2.0",
    "golden": "target_chart_iso12233.png"
  },
  
  "test_scene": {
    "type": "traffic_light",
    "description": "交通灯红绿灯还原测试",
    "image_path": "/test_images/traffic_01.jpg"
  },
  
  "results": {
    "traffic_light_color": {
      "red": {
        "current_score": 88,
        "reference_score": 82,
        "delta": +6,
        "status": "improved"
      },
      "green": {
        "current_score": 92,
        "reference_score": 91,
        "delta": +1,
        "status": "similar"
      }
    },
    "contour": {
      "sharpness_score": {
        "current": 78,
        "reference": 75,
        "delta": +3
      }
    }
  },
  
  "overall_assessment": {
    "status": "improved",
    "summary": "ISP_v2.1 在交通灯颜色还原上整体改善，红色还原提升明显",
    "recommendations": [
      "继续保持当前 CCM 配置",
      "绿色还原已达满意水平，可适当降低饱和度"
    ]
  },
  
  "visualizations": [
    "/reports/ISP_COMP_001/diff_heatmap.png",
    "/reports/ISP_COMP_001/contour_comparison.png"
  ]
}
```

### 7.2 报告可视化

```
┌─────────────────────────────────────────────────────────────────┐
│              ISP Version A vs B 对比报告                          │
│              生成时间: 2026-03-28 10:30                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │ ISP_v2.0   │  │ ISP_v2.1   │  │ Difference  │           │
│  │             │  │             │  │             │           │
│  │   [图像]   │  │   [图像]   │  │  [差异图]  │           │
│  │             │  │             │  │             │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     指标对比                              │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  维度          │  v2.0  │  v2.1  │  变化  │  状态       │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  红灯还原     │  82    │  88    │  +6    │  ✅ 改善   │  │
│  │  绿灯还原     │  91    │  92    │  +1    │  ✅ 持平   │  │
│  │  锐度评分     │  75    │  78    │  +3    │  ✅ 改善   │  │
│  │  噪声控制     │  85    │  83    │  -2    │  ⚠️ 略降   │  │
│  │  色彩准确     │  88    │  90    │  +2    │  ✅ 改善   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     AI 分析建议                           │  │
│  │                                                           │  │
│  │  综合评估: ISP_v2.1 整体优于 v2.0，尤其在交通灯还原     │  │
│  │  方面改善明显。噪声控制略有下降，建议微调降噪参数。       │  │
│  │                                                           │  │
│  │  建议:                                                     │  │
│  │  1. 保持当前 CCM 配置，继续优化红色还原                    │  │
│  │  2. 可适当提高 Luma NR 强度，补偿噪声略降                 │  │
│  │  3. 考虑在 v2.2 中加入交通灯专用优化模块                   │  │
│  │                                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 八、已确认的格式细节

### ✅ Comp12 格式

| 属性 | 值 |
|------|-----|
| 文件头 | 无 |
| 大小端 | 小端 (LSB) |
| 分辨率 | 3840×2160 或 1920×1080 |
| Bit Depth | 12-bit |

### ✅ CModel ISP

| 属性 | 值 |
|------|-----|
| 现有代码 | C/C++ 实现 |
| 输入 | 纯 RAW（自动解析 Pattern） |
| 输出 | RGB 24bit |
| 调用方式 | 待定（subprocess/ctypes/Python 集成） |

### ✅ 交通灯检测两个场景

| 场景 | 说明 | 评估方式 |
|------|------|----------|
| **场景1** | ISP 参数调优 | 客观指标评估（颜色/轮廓） |
| **场景2** | ADAS 感知验证 | 感知模型检测结果 |

### ✅ 阈值标准

| 指标 | 标准 | 参考 |
|------|------|------|
| 交通灯色差 | 按业内标准 | IEC 61888 |
| 其他指标 | 按行业通用标准 | ISO 12233 等 |

---

## 九、模块依赖

### 9.1 Python 依赖

```txt
# Core
opencv-python>=4.8.0
numpy>=1.24.0

# Metrics
scikit-image>=0.21.0  # SSIM
# BRISQUE (可从 skimage 或单独安装)
# pip install brisque

# Deep Learning (可选, 用于 LPIPS)
# pip install lpips torch

# Report
jinja2>=3.1.0  # 报告模板
markdown>=3.4.0
```

### 9.2 与 OpenClaw 集成

```
ISP Version Comparator
        │
        ├── Skill: isp_version_compare
        │       └── SKILL.md
        │
        ├── Tool: isp_compare
        │       └── comparsion.py (Python)
        │
        └── Memory: 对比历史记录
```

---

## 十、开发计划建议

### Phase 1: 基础框架 (1周)

| 任务 | 交付物 |
|------|--------|
| Comp12 解析器 | 能读取 comp12 文件 |
| CModel ISP 基础实现 | Demosaic + 基础处理 |
| 图像导入 | 支持 JPEG/PNG/TIFF |
| 基础对比 | PSNR/SSIM 计算 |

### Phase 2: 核心功能 (1-2周)

| 任务 | 交付物 |
|------|--------|
| 交通灯检测 + 评估 | 自动/手动 ROI |
| 轮廓评估 | MTF/锐度/边缘完整性 |
| 多维度指标计算 | 清晰度/噪声/色彩 |
| 对比报告生成 | Markdown + 可视化 |

### Phase 3: 智能化 (持续)

| 任务 | 交付物 |
|------|--------|
| AI 分析建议 | 基于结果给出调参方向 |
| 历史对比追踪 | 版本迭代对比曲线 |
| 自动调参建议 | 结合知识库给出建议 |

---

*文档状态: v1.0 草稿*
*待石先生确认格式细节后完善*
*创建日期: 2026-03-28*

---

## 十一、已确认的交付要求

### ✅ 感知模型

| 属性 | 说明 |
|------|------|
| **状态** | 预留接口，最后阶段接入 |
| **接口要求** | 标准接口，支持 YOLO 等主流模型 |
| **接入时机** | Phase 3 或最后 |

### ✅ Golden Reference

| 属性 | 说明 |
|------|------|
| **数据来源** | 对应的实景都有 Golden 图片 |
| **用途** | 作为对比基准 |
| **格式** | RGB 输出图像（JPEG/PNG） |

### ✅ 报告格式

| 格式 | 说明 |
|------|------|
| **PDF** | 主要导出格式，用于存档和分享 |

---

## 十二、技术方案概要

### 12.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                  ISP 版本对比技术架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  输入层                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Comp12   │  │ Golden   │  │ ISP v1/v2│              │
│  │ RAW      │  │ Images   │  │ Output   │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │             │             │                       │
│       └─────────────┼─────────────┘                       │
│                     │                                      │
│                     ▼                                      │
│  ┌─────────────────────────────────────────┐              │
│  │          Python 分析层                    │              │
│  │  • Comp12 Parser (小端, 无文件头)       │              │
│  │  • CModel ISP 调用 (C/C++)              │              │
│  │  • 颜色/轮廓/锐度指标                   │              │
│  │  • 感知模型接口 (预留)                   │              │
│  └────────────────────┬────────────────────┘              │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────┐              │
│  │          报告层                          │              │
│  │  • PDF 报告生成                          │              │
│  │  • AI 调参建议                          │              │
│  │  • 历史记录                              │              │
│  └─────────────────────────────────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 12.2 数据流

```
1. 输入 Comp12 RAW + Pattern
       ↓
2. 调用 CModel ISP (C/C++) → RGB 24bit
       ↓
3. 与 Golden Image 对比
       ↓
4. 计算各维度指标
       ↓
5. 生成对比报告 (PDF)
       ↓
6. AI 分析建议
       ↓
7. 记录到历史
```

### 12.3 PDF 报告结构

```
┌─────────────────────────────────────────────────────────────┐
│                    ISP 版本对比报告                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  报告信息                                                    │
│  • 版本: ISP_v2.1 vs ISP_v2.0                             │
│  • 日期: 2026-03-28                                       │
│  • 测试场景: 交通灯还原                                     │
│                                                              │
│  视觉对比                                                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                   │
│  │ Golden │  │ ISP v1 │  │ ISP v2 │                   │
│  │        │  │        │  │        │                   │
│  └─────────┘  └─────────┘  └─────────┘                   │
│                                                              │
│  指标对比表                                                   │
│  ┌────────────────┬────────┬────────┬────────┐            │
│  │ 指标           │ Golden │ ISP v1 │ ISP v2 │            │
│  ├────────────────┼────────┼────────┼────────┤            │
│  │ 红灯色差 Δuv   │  -     │ 0.032  │ 0.018  │            │
│  │ 绿灯色差 Δuv   │  -     │ 0.021  │ 0.019  │            │
│  │ 锐度评分       │  85    │  78    │  82    │            │
│  └────────────────┴────────┴────────┴────────┘            │
│                                                              │
│  结论与建议                                                  │
│  • ISP_v2.1 在交通灯颜色还原上优于 v2.0                   │
│  • 建议: 继续保持当前 CCM 配置                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

*文档状态: v1.2 完善中*
*创建日期: 2026-03-28*
*最后更新: 2026-03-28*
