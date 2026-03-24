FROM python:3.9-slim

LABEL maintainer="ISP-Agent Team"
LABEL description="ISP Image/Video Processing Assistant"

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY src/ ./src/
COPY config/ ./config/

# 创建输出目录
RUN mkdir -p /app/outputs /app/logs

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]

# 备用: 运行CLI
# CMD ["python", "-m", "src.cli_enhanced", "--help"]
