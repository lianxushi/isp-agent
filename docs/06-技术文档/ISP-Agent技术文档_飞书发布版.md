# ISP-Agent 技术文档

> 生成时间：2026-03-21

---

## 一、技术架构

### 1.1 系统架构（本地Python + API大脑）

用户接口层 → 应用层 → 能力层(API) → 工具层

### 1.2 核心技术栈

- Python 3.9+ / OpenCV / NumPy / Pillow
- MiniMax API (默认) / OpenAI / Anthropic

### 1.3 项目结构

```
isp-agent/
├── src/
│   ├── main.py          # CLI入口
│   ├── agent/           # LLM客户端 + 问答引擎
│   ├── tools/           # 图像/视频分析器
│   └── utils/           # 配置 + 日志
├── config/              # 配置文件
└── venv/               # 虚拟环境
```

---

## 二、已实现功能

### 2.1 图像分析 ✅
- 基本信息（分辨率/格式/大小）
- 直方图分析（RGB + 灰度）
- 动态范围 / 噪声分析
- 亮度/对比度 / 色彩分析
- EXIF解析

### 2.2 ISP知识问答 ✅
- ISP模块原理（降噪/AWB/CCM等）
- 图像问题诊断
- 调优建议
- 多轮对话

### 2.3 视频分析 ✅
- 基本信息 / 码率 / 音频
- 抽帧功能

### 2.4 图像处理 ✅
- 降噪 / 锐化 / 色彩调整

---

## 三、存在缺陷

| 缺陷 | 程度 | 说明 |
|------|------|------|
| RAW格式支持不完整 | 中 | 仅EXIF，无DNG处理 |
| 视频处理功能少 | 中 | 仅有基本信息 |
| 无Web界面 | 低 | 仅有CLI |
| 无API服务 | 低 | 仅本地运行 |

---

## 四、后续开发计划

### 短期（1-2周）
- 完善图像分析（Gamma/边缘）
- RAW格式支持
- 错误处理优化

### 中期（1个月）
- 视频质量分析
- Web界面
- API服务

### 长期（季度）
- 多Agent协作
- 移动端支持
- 云端部署

---

## 五、使用说明

```bash
cd ~/.openclaw/workspace/isp-agent
source venv/bin/activate
export MINIMAX_API_KEY=your-key

# 对话
python src/main.py chat
# 分析图像
python src/main.py analyze image.jpg
# 分析视频
python src/main.py analyze video.mp4
```

---

## 六、总结

MVP版本已完成：
- ✅ 图像分析
- ✅ ISP知识问答
- ✅ 视频分析
- ✅ 图像处理

*本文档由 OpenClaw 自动生成*
