#!/usr/bin/env python3
"""
ISP-Agent 命令行增强入口
整合所有分析功能
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from src.tools.image_analyzer import ImageAnalyzer
from src.tools.video_analyzer import VideoAnalyzer
from src.tools.hdr_processor import HDRProcessor
from src.tools.automotive_analyzer import AutomotiveQualityAnalyzer
from src.tools.ai_quality_scorer import AIQualityScorer
from src.tools.tuning_knowledge import ISPTuningKnowledge
from src.tools.pipeline_visualizer import ISPPipelineVisualizer
from src.tools.raw_processor import RawProcessor


def cmd_analyze(args):
    """分析图像"""
    analyzer = ImageAnalyzer()
    result = analyzer.analyze(args.image)
    
    if args.json:
        print(result.to_json())
    else:
        print("=" * 50)
        print(f"📷 图像分析结果: {result.file_name}")
        print("=" * 50)
        print(f"📐 分辨率: {result.width} x {result.height}")
        print(f"📁 格式: {result.format}")
        print(f"💾 大小: {result.size_kb:.1f} KB")
        
        if result.dynamic_range:
            dr = result.dynamic_range
            print(f"\n🔆 动态范围: {dr['useful_range']} (min:{dr['min']}, max:{dr['max']})")
        
        if result.noise_level:
            print(f"🔊 噪声水平: {result.noise_level:.1f}")
        
        if result.color_analysis:
            ca = result.color_analysis
            print(f"\n🎨 色彩分析:")
            print(f"   白平衡: {ca.get('white_balance', 'N/A')}")
            print(f"   饱和度: {ca.get('saturation', 'N/A')}")
        
        if result.exif:
            print(f"\n📸 EXIF信息:")
            for k, v in result.exif.items():
                print(f"   {k}: {v}")
        
        if args.auto_tune:
            knowledge = ISPTuningKnowledge()
            tuning = knowledge.generate_tuning_suggestions(result.to_dict(), 'automotive')
            print(f"\n🔧 调优建议:")
            print(f"   {tuning['overall_assessment']}")
            for rec in tuning['diagnosis'].get('recommendations', [])[:2]:
                print(f"   - {rec.get('issue', '')}")
                for sol in rec.get('solutions', [])[:2]:
                    print(f"     → {sol}")


def cmd_automotive(args):
    """车载场景分析"""
    analyzer = AutomotiveQualityAnalyzer()
    
    resolution = None
    if args.resolution:
        w, h = map(int, args.resolution.split('x'))
        resolution = (w, h)
    
    result = analyzer.analyze(
        args.image,
        scene_type=args.scene,
        resolution=resolution,
        fps=args.fps,
        fov=args.fov
    )
    
    if args.json:
        print(json.dumps({
            'overall_score': result.overall_score,
            'night_vision_score': result.night_vision_score,
            'hdr_score': result.hdr_score,
            'motion_blur_score': result.motion_blur_score,
            'recommendations': result.recommendations,
        }, ensure_ascii=False, indent=2))
    else:
        print("=" * 50)
        print(f"🚗 车载场景分析: {args.scene}")
        print("=" * 50)
        print(f"⭐ 综合评分: {result.overall_score}/100")
        print(f"\n📊 分项评分:")
        print(f"   夜景成像: {result.night_vision_score}")
        print(f"   HDR效果: {result.hdr_score}")
        print(f"   运动模糊: {result.motion_blur_score}")
        print(f"\n💡 建议:")
        for rec in result.recommendations:
            print(f"   • {rec}")


def cmd_quality(args):
    """AI质量评分"""
    scorer = AIQualityScorer()
    result = scorer.score(args.image)
    
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("=" * 50)
        print(f"🤖 AI质量评分")
        print("=" * 50)
        print(f"⭐ MOS预测: {result['mos_predicted']}/5.0 ({result['mos_description']})")
        print(f"📊 综合评分: {result['overall']}/100")
        print(f"🏷️ 等级: {result['grade']}")
        print(f"\n📈 分项评分:")
        print(f"   清晰度: {result['sharpness_score']} - {result['details']['sharpness']}")
        print(f"   噪声: {result['noise_score']} - {result['details']['noise']}")
        print(f"   伪影: {result['artifact_score']} - {result['details']['artifact']}")
        print(f"   色彩: {result['color_score']} - {result['details']['color']}")


def cmd_pipeline(args):
    """ISP Pipeline可视化"""
    viz = ISPPipelineVisualizer()
    
    if args.explain:
        print(viz.explain_stage(args.explain))
    elif args.config:
        config = viz.generate_config(args.config)
        print(json.dumps(config, ensure_ascii=False, indent=2))
    else:
        print(viz.visualize(args.type))


def cmd_video(args):
    """视频分析"""
    analyzer = VideoAnalyzer()
    
    if args.extract_frames:
        frames = analyzer.extract_frames(
            args.video,
            args.output_dir or 'frames',
            interval=args.interval,
            max_frames=args.max_frames
        )
        print(f"✅ 已抽取 {len(frames)} 帧到 {args.output_dir or 'frames'}")
    else:
        info = analyzer.get_info(args.video)
        
        if args.json:
            print(info.to_json())
        else:
            print("=" * 50)
            print(f"🎬 视频信息: {info.file_name}")
            print("=" * 50)
            print(f"📐 分辨率: {info.width} x {info.height}")
            print(f"🎞️ 帧率: {info.fps} fps")
            print(f"⏱️ 时长: {info.duration_seconds:.1f} 秒")
            print(f"🎨 编码: {info.codec}")
            print(f"💾 大小: {info.size_mb:.1f} MB")


def cmd_raw(args):
    """RAW处理"""
    processor = RawProcessor()
    
    if args.info:
        info = processor.get_info(args.raw)
        print(json.dumps(info, ensure_ascii=False, indent=2))
    elif args.to_tiff:
        result = processor.to_tiff(args.raw, args.output or args.raw.replace('.raw', '.tiff'))
        print(f"✅ 已转换为TIFF: {result.get('output')}")
    else:
        result = processor.get_info(args.raw)
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description='ISP-Agent 命令行工具')
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # analyze 命令
    p_analyze = subparsers.add_parser('analyze', help='分析图像')
    p_analyze.add_argument('image', help='图像路径')
    p_analyze.add_argument('--json', action='store_true', help='JSON输出')
    p_analyze.add_argument('--auto-tune', action='store_true', help='自动调优建议')
    p_analyze.set_defaults(func=cmd_analyze)
    
    # automotive 命令
    p_auto = subparsers.add_parser('automotive', help='车载场景分析')
    p_auto.add_argument('image', help='图像路径')
    p_auto.add_argument('--scene', default='adas_front', choices=['adas_front', 'surround', 'dms'], help='场景类型')
    p_auto.add_argument('--resolution', help='分辨率 (如 1920x1080)')
    p_auto.add_argument('--fps', type=float, help='帧率')
    p_auto.add_argument('--fov', type=float, help='视场角')
    p_auto.add_argument('--json', action='store_true', help='JSON输出')
    p_auto.set_defaults(func=cmd_automotive)
    
    # quality 命令
    p_quality = subparsers.add_parser('quality', help='AI质量评分')
    p_quality.add_argument('image', help='图像路径')
    p_quality.add_argument('--json', action='store_true', help='JSON输出')
    p_quality.set_defaults(func=cmd_quality)
    
    # pipeline 命令
    p_pipe = subparsers.add_parser('pipeline', help='ISP Pipeline')
    p_pipe.add_argument('--type', default='standard', choices=['standard', 'simple', 'advanced'])
    p_pipe.add_argument('--explain', help='解释特定模块')
    p_pipe.add_argument('--config', help='生成场景配置')
    p_pipe.set_defaults(func=cmd_pipeline)
    
    # video 命令
    p_video = subparsers.add_parser('video', help='视频分析')
    p_video.add_argument('video', help='视频路径')
    p_video.add_argument('--json', action='store_true', help='JSON输出')
    p_video.add_argument('--extract-frames', action='store_true', help='抽取帧')
    p_video.add_argument('--output-dir', help='输出目录')
    p_video.add_argument('--interval', type=int, default=1, help='抽帧间隔')
    p_video.add_argument('--max-frames', type=int, default=100, help='最大帧数')
    p_video.set_defaults(func=cmd_video)
    
    # raw 命令
    p_raw = subparsers.add_parser('raw', help='RAW处理')
    p_raw.add_argument('raw', help='RAW文件路径')
    p_raw.add_argument('--info', action='store_true', help='显示信息')
    p_raw.add_argument('--to-tiff', action='store_true', help='转换为TIFF')
    p_raw.add_argument('--output', help='输出路径')
    p_raw.set_defaults(func=cmd_raw)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
