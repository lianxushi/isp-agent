FROM python:3.9-slim

LABEL maintainer="ISP-Agent Team"
LABEL description="ISP Image/Video Processing Assistant - v0.4"

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
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY src/ ./src/
COPY config/ ./config/
COPY docs/ ./docs/

# 创建输出和数据目录
RUN mkdir -p /app/outputs /app/logs /app/data

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# 暴露端口
# 5000: Flask API
# 8080: Alternative API
EXPOSE 5000 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health', timeout=5)" || exit 1

# 启动命令 - Flask API
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]

# 备用: 运行CLI
# CMD ["python", "-m", "src.cli_enhanced", "--help"]
