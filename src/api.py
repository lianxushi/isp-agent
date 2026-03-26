#!/usr/bin/env python3
"""
ISP-Agent REST API 服务
基于Flask的Web服务
"""
import os
import io
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# 添加src目录到路径
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.tools.image_analyzer import ImageAnalyzer
from src.tools.video_analyzer import VideoAnalyzer
from src.tools.hdr_processor import HDRProcessor
from src.tools.automotive_analyzer import AutomotiveQualityAnalyzer
from src.tools.ai_quality_scorer import AIQualityScorer
from src.tools.tuning_knowledge import ISPTuningKnowledge
from src.tools.pipeline_visualizer import ISPPipelineVisualizer
from src.tools.raw_processor import RawProcessor
from src.tools.batch_processor import BatchProcessor
from src.tools.enhanced_qa import EnhancedQAEngine

app = Flask(__name__)
CORS(app)


def _save_upload():
    """保存上传文件到临时目录"""
    if 'file' not in request.files:
        return None, jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    suffix = os.path.splitext(file.filename)[1].lower()
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    file.save(temp_path)
    return temp_path, None, None


def _cleanup(path):
    """清理临时文件"""
    if path and os.path.exists(path):
        os.remove(path)


# 加载配置
_config = None


def _get_config():
    global _config
    if _config is None:
        try:
            _config = load_config()
        except Exception:
            _config = None
    return _config


# 初始化分析器（延迟初始化）
_analyzers = {}


def _get_analyzer(name, factory):
    if name not in _analyzers:
        config = _get_config()
        _analyzers[name] = factory(config.config if config else None)
    return _analyzers[name]


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'ok', 'service': 'isp-agent-api', 'version': '0.3.0'})


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """分析图像"""
    temp_path, error, status = _save_upload()
    if error:
        return error, status
    try:
        analyzer = _get_analyzer('image', lambda c: ImageAnalyzer(c))
        result = analyzer.analyze(temp_path)
        return jsonify(result.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        _cleanup(temp_path)


@app.route('/api/video/analyze', methods=['POST'])
def analyze_video():
    """分析视频"""
    temp_path, error, status = _save_upload()
    if error:
        return error, status
    try:
        analyzer = _get_analyzer('video', lambda c: VideoAnalyzer(c))
        info = analyzer.get_info(temp_path)
        return jsonify(info.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        _cleanup(temp_path)


@app.route('/api/video/extract_frames', methods=['POST'])
def extract_video_frames():
    """抽取视频帧"""
    temp_path, error, status = _save_upload()
    if error:
        return error, status
    try:
        interval = request.form.get('interval', type=int, default=1)
        max_frames = request.form.get('max_frames', type=int, default=100)
        output_dir = request.form.get('output_dir') or None
        
        analyzer = _get_analyzer('video', lambda c: VideoAnalyzer(c))
        frames = analyzer.extract_frames(temp_path, output_dir or 'frames', 
                                         interval=interval, max_frames=max_frames)
        return jsonify({'frames': frames, 'count': len(frames)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        _cleanup(temp_path)


@app.route('/api/hdr/merge', methods=['POST'])
def merge_hdr():
    """多帧HDR合成"""
    files = request.files.getlist('files')
    if not files or len(files) < 2:
        return jsonify({'error': 'At least 2 images required for HDR'}), 400
    
    temp_paths = []
    for f in files:
        fd, path = tempfile.mkstemp(suffix=os.path.splitext(f.filename)[1].lower())
        os.close(fd)
        f.save(path)
        temp_paths.append(path)
    
    output_path = tempfile.mktemp(suffix='.jpg')
    
    try:
        processor = _get_analyzer('hdr', lambda c: HDRProcessor())
        result = processor.merge_hdr(temp_paths, output_path)
        return jsonify({'output': output_path, 'details': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        for p in temp_paths:
            _cleanup(p)


@app.route('/api/automotive', methods=['POST'])
def analyze_automotive():
    """车载场景分析"""
    temp_path, error, status = _save_upload()
    if error:
        return error, status
    try:
        scene_type = request.form.get('scene_type', 'adas_front')
        
        analyzer = _get_analyzer('automotive', lambda c: AutomotiveQualityAnalyzer())
        result = analyzer.analyze(
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
        _cleanup(temp_path)


@app.route('/api/quality', methods=['POST'])
def score_quality():
    """AI质量评分"""
    temp_path, error, status = _save_upload()
    if error:
        return error, status
    try:
        analyzer = _get_analyzer('quality', lambda c: AIQualityScorer())
        result = analyzer.score(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        _cleanup(temp_path)


@app.route('/api/tune', methods=['POST'])
def get_tuning():
    """获取调优建议"""
    data = request.json
    if not data or 'analysis' not in data:
        return jsonify({'error': 'No analysis data provided'}), 400
    try:
        scene_type = data.get('scene_type', 'automotive')
        analyzer = _get_analyzer('tuning', lambda c: ISPTuningKnowledge())
        result = analyzer.generate_tuning_suggestions(data['analysis'], scene_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pipeline', methods=['GET'])
def get_pipeline():
    """获取ISP Pipeline"""
    pipeline_type = request.args.get('type', 'standard')
    try:
        viz = _get_analyzer('pipeline', lambda c: ISPPipelineVisualizer())
        result = viz.visualize(pipeline_type)
        return jsonify({'pipeline': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pipeline/modules/<module_name>', methods=['GET'])
def get_module_info(module_name):
    """获取特定模块信息"""
    try:
        viz = _get_analyzer('pipeline', lambda c: ISPPipelineVisualizer())
        info = viz.get_module_info(module_name)
        if info:
            return jsonify(info)
        return jsonify({'error': 'Module not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/raw/info', methods=['POST'])
def raw_info():
    """获取RAW文件信息"""
    temp_path, error, status = _save_upload()
    if error:
        return error, status
    try:
        processor = _get_analyzer('raw', lambda c: RawProcessor())
        info = processor.get_info(temp_path)
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        _cleanup(temp_path)


@app.route('/api/raw/to_tiff', methods=['POST'])
def raw_to_tiff():
    """RAW转TIFF"""
    temp_path, error, status = _save_upload()
    if error:
        return error, status
    output_path = request.form.get('output') or temp_path.replace(
        os.path.splitext(temp_path)[1], '.tiff')
    try:
        processor = _get_analyzer('raw', lambda c: RawProcessor())
        result = processor.to_tiff(temp_path, output_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        _cleanup(temp_path)


@app.route('/api/batch/analyze', methods=['POST'])
def batch_analyze():
    """批量分析图像"""
    data = request.json
    if not data or 'directory' not in data:
        return jsonify({'error': 'directory required'}), 400
    try:
        processor = _get_analyzer('batch', lambda c: BatchProcessor())
        result = processor.analyze_batch(data['directory'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/qa/ask', methods=['POST'])
def qa_ask():
    """增强问答"""
    data = request.json
    if not data or 'question' not in data:
        return jsonify({'error': 'question required'}), 400
    try:
        qa = _get_analyzer('enhanced_qa', lambda c: EnhancedQAEngine())
        result = qa.ask(data['question'], data.get('context'))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_api(host='0.0.0.0', port=5000, debug=False):
    """运行API服务"""
    print(f"Starting ISP-Agent API on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_api(debug=True)
