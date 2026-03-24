#!/usr/bin/env python3
"""
ISP-Agent REST API 服务
基于Flask的Web服务
"""
import os
import io
import base64
from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np

# 导入分析模块
from tools.image_analyzer import ImageAnalyzer
from tools.video_analyzer import VideoAnalyzer
from tools.automotive_analyzer import AutomotiveQualityAnalyzer
from tools.ai_quality_scorer import AIQualityScorer
from tools.tuning_knowledge import ISPTuningKnowledge
from tools.pipeline_visualizer import ISPPipelineVisualizer
from tools.export_manager import ExportManager


app = Flask(__name__)

# 初始化分析器
image_analyzer = ImageAnalyzer()
video_analyzer = VideoAnalyzer()
auto_analyzer = AutomotiveQualityAnalyzer()
quality_scorer = AIQualityScorer()
knowledge = ISPTuningKnowledge()
pipeline_viz = ISPPipelineVisualizer()
export_mgr = ExportManager()


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'ok', 'service': 'isp-agent-api'})


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """分析图像"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # 保存临时文件
    temp_path = f'/tmp/{file.filename}'
    file.save(temp_path)
    
    try:
        result = image_analyzer.analyze(temp_path)
        return jsonify(result.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/api/automotive', methods=['POST'])
def analyze_automotive():
    """车载场景分析"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    scene_type = request.form.get('scene_type', 'adas_front')
    
    temp_path = f'/tmp/{file.filename}'
    file.save(temp_path)
    
    try:
        result = auto_analyzer.analyze(
            temp_path,
            scene_type=scene_type,
            resolution=None,
            fps=None,
            fov=None
        )
        
        return jsonify({
            'overall_score': result.overall_score,
            'night_vision_score': result.night_vision_score,
            'hdr_score': result.hdr_score,
            'motion_blur_score': result.motion_blur_score,
            'recommendations': result.recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/api/quality', methods=['POST'])
def score_quality():
    """AI质量评分"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    temp_path = f'/tmp/{file.filename}'
    file.save(temp_path)
    
    try:
        result = quality_scorer.score(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/api/tune', methods=['POST'])
def get_tuning():
    """获取调优建议"""
    data = request.json
    
    if not data or 'analysis' not in data:
        return jsonify({'error': 'No analysis data provided'}), 400
    
    try:
        scene_type = data.get('scene_type', 'automotive')
        result = knowledge.generate_tuning_suggestions(data['analysis'], scene_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pipeline', methods=['GET'])
def get_pipeline():
    """获取ISP Pipeline"""
    pipeline_type = request.args.get('type', 'standard')
    
    try:
        viz = pipeline_viz.visualize(pipeline_type)
        return jsonify({'pipeline': viz})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pipeline/modules/<module_name>', methods=['GET'])
def get_module_info(module_name):
    """获取特定模块信息"""
    try:
        info = pipeline_viz.get_module_info(module_name)
        if info:
            return jsonify(info)
        return jsonify({'error': 'Module not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_api(host='0.0.0.0', port=5000, debug=False):
    """运行API服务"""
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_api(debug=True)
