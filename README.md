# ISP-Agent
基于LLM的ISP图像/视频处理助手 (本地Python + API大脑)

## 项目简介

ISP-Agent 是一款基于**本地Python运行时 + 远程API大脑**架构的ISP图像/视频处理助手。

### 核心架构

```
┌─────────────────┐      API调用       ┌─────────────────┐
│   本地Python    │ ◄────────────────► │   LLM API       │
│   (图像处理)    │    HTTP请求/响应   │  (MiniMax/GPT)  │
└─────────────────┘                    └─────────────────┘
```

- 🖥️ **本地处理**：图像/视频分析在本地执行，保护隐私
- 🧠 **云端智能**：复杂推理由LLM API提供
- 🔧 **灵活切换**：支持MiniMax/OpenAI/Anthropic

## 功能特性

- 📊 图像质量分析 (直方图、噪声、动态范围、色彩)
- 🔍 ISP参数诊断 (EXIF、DNG元数据)
- 🎬 视频处理分析 (基本信息、抽帧、质量检测)
- 💡 智能调优建议 (基于分析结果的LLM建议)
- 🗣️ 自然语言对话 (中文ISP领域问答)

## 快速开始

### 1. 环境要求

- Python 3.9+
- macOS / Linux / Windows

### 2. 安装

```bash
# 克隆项目
git clone https://github.com/your-repo/isp-agent.git
cd isp-agent

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置API密钥

```bash
# MiniMax (默认，推荐)
export MINIMAX_API_KEY="your-api-key"

# 或 OpenAI
# export OPENAI_API_KEY="sk-xxx"

# 或 Anthropic
# export ANTHROPIC_API_KEY="sk-ant-xxx"
```

### 4. 运行

```bash
# 对话模式
python src/main.py --chat

# 分析图像
python src/main.py --image your_image.jpg

# 分析视频
python src/main.py --video your_video.mp4
```

## 项目结构

```
isp-agent/
├── docs/                    # 项目文档
│   ├── 00-项目管理/        # 项目计划
│   ├── 01-需求分析/        # 需求规格说明
│   ├── 02-架构设计/        # 系统架构
│   ├── 03-详细设计/        # 详细设计
│   ├── 04-测试计划/        # 测试计划
│   └── 05-用户手册/        # 用户手册
├── src/                     # 源代码
│   ├── main.py              # 入口文件
│   ├── llm_client.py        # LLM客户端 (API调用)
│   ├── image_analyzer.py    # 图像分析器
│   └── qa_engine.py         # 问答引擎
├── config/                  # 配置文件
│   └── default.yaml         # 默认配置
├── logs/                    # 日志目录
└── requirements.txt         # Python依赖
```

## 文档

- [需求规格说明书](docs/01-需求分析/需求规格说明书.md)
- [系统架构设计](docs/02-架构设计/系统架构设计.md)
- [详细设计文档](docs/03-详细设计/详细设计文档.md)
- [项目计划](docs/00-项目管理/项目计划.md)

## 技术栈

| 类别 | 技术 |
|------|------|
| 本地运行时 | Python 3.9+ |
| 图像处理 | OpenCV, Pillow, rawpy |
| 视频处理 | FFmpeg, opencv-python |
| LLM API | MiniMax (默认), OpenAI, Anthropic |

## 许可证

MIT License
