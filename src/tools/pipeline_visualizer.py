#!/usr/bin/env python3
"""
ISP Pipeline 可视化模块
展示ISP处理流程和数据流
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from ..utils.logger import setup_logger

logger = setup_logger('isp-agent.pipeline')


@dataclass
class ISPPipelineStage:
    """ISP Pipeline 阶段"""
    name: str
    description: str
    enabled: bool
    params: Dict[str, Any]


class ISPPipelineVisualizer:
    """
    ISP Pipeline 可视化器
    
    标准ISP Pipeline:
    Bypass → Demosaic → Denoise → Sharpening → Color Correction → 
    Gamma → Tone Mapping → CSC → Output
    """
    
    # 标准ISP模块定义
    STANDARD_PIPELINE = [
        ISPPipelineStage('Bayer', 'Bayer阵列原始数据', True, {}),
        ISPPipelineStage('Demosaic', '去马赛克/插值', True, {'method': 'malvar'}),
        ISPPipelineStage('LSC', '镜头阴影校正', True, {'strength': 1.0}),
        ISPPipelineStage('AWB', '自动白平衡', True, {'mode': 'auto'}),
        ISPPipelineStage('Denoise', '降噪', True, {'strength': 1.0, 'method': 'bilateral'}),
        ISPPipelineStage('Sharpening', '锐化', True, {'strength': 1.0, 'radius': 1.0}),
        ISPPipelineStage('CCM', '色彩校正矩阵', True, {}),
        ISPPipelineStage('Gamma', 'Gamma校正', True, {'value': 2.2}),
        ISPPipelineStage('LTM', '局部色调映射', False, {'strength': 0.5}),
        ISPPipelineStage('Contrast', '对比度调整', True, {'value': 1.0}),
        ISPPipelineStage('Saturation', '饱和度调整', True, {'value': 1.0}),
        ISPPipelineStage('CSC', '色彩空间转换', True, {'to': 'sRGB'}),
    ]
    
    # 模块参数说明
    MODULE_PARAMS = {
        'Demosaic': {
            'malvar': 'Malvar-He-Cutler算法(默认)',
            'menon': 'Menon算法',
            'nearest': '最近邻(快速但质量低)',
        },
        'Denoise': {
            'bilateral': '双边滤波(保边)',
            'nlm': '非局部均值(高质量)',
            'gaussian': '高斯滤波(快速)',
            'median': '中值滤波(去除椒盐噪声)',
        },
        'Sharpening': {
            'unsharp_mask': 'USM锐化(默认)',
            'laplacian': '拉普拉斯锐化',
            'edge': '边缘增强',
        },
        'AWB': {
            'auto': '自动白平衡',
            'gray_world': '灰度世界假设',
            'white_patch': '白块假设',
        },
    }
    
    def __init__(self):
        pass
    
    def get_pipeline(self, pipeline_type: str = 'standard') -> List[ISPPipelineStage]:
        """获取ISP Pipeline配置"""
        if pipeline_type == 'standard':
            return self.STANDARD_PIPELINE
        elif pipeline_type == 'simple':
            return [s for s in self.STANDARD_PIPELINE if s.name in ['Demosaic', 'AWB', 'Gamma', 'CSC']]
        elif pipeline_type == 'advanced':
            return self.STANDARD_PIPELINE
        else:
            return self.STANDARD_PIPELINE
    
    def visualize(self, pipeline_type: str = 'standard') -> str:
        """生成Pipeline可视化文本"""
        pipeline = self.get_pipeline(pipeline_type)
        
        lines = ["═" * 50]
        lines.append("📷 ISP Pipeline 数据流")
        lines.append("═" * 50)
        
        for i, stage in enumerate(pipeline):
            status = "✅" if stage.enabled else "❌"
            lines.append(f"{status} [{i+1:2d}] {stage.name:15s} - {stage.description}")
            
            if stage.params:
                for k, v in stage.params.items():
                    lines.append(f"       └─ {k}: {v}")
        
        lines.append("═" * 50)
        lines.append("➡️  输出图像")
        
        return "\n".join(lines)
    
    def get_module_info(self, module_name: str) -> Optional[Dict[str, Any]]:
        """获取模块详细信息"""
        for stage in self.STANDARD_PIPELINE:
            if stage.name == module_name:
                return {
                    'name': stage.name,
                    'description': stage.description,
                    'params': stage.params,
                    'available_params': self.MODULE_PARAMS.get(module_name, {}),
                }
        return None
    
    def explain_stage(self, stage_name: str) -> str:
        """解释特定ISP阶段的作用"""
        explanations = {
            'Bayer': '''
📷 Bayer (拜耳阵列)
   作用: 相机传感器原始数据格式
   说明: 大多数CMOS传感器使用拜耳色彩滤镜阵列，
         每个像素只记录R/G/B中的一种颜色
            ''',
            'Demosaic': '''
🔄 Demosaic (去马赛克)
   作用: 从Bayer数据重建完整RGB图像
   常用算法: Malvar, Menon, 最近邻
   质量影响: 算法选择影响细节和色彩准确性
            ''',
            'LSC': '''
🌑 LSC (镜头阴影校正)
   作用: 补偿镜头边缘亮度衰减
   类型: 径向校正/平面校正
   参数: 校正强度 (0-2.0)
            ''',
            'AWB': '''
🌡️ AWB (自动白平衡)
   作用: 消除不同光源下的色偏
   算法: 灰度世界、白块、机器学习
   场景: 室内/室外/夜景等不同模式
            ''',
            'Denoise': '''
🔊 Denoise (降噪)
   作用: 减少图像噪声
   方法: 
   - 双边滤波: 保边降噪
   - NLM: 高质量但慢
   - 高斯: 快速但模糊
   参数: 降噪强度、窗口大小
            ''',
            'Sharpening': '''
🔪 Sharpening (锐化)
   作用: 增强边缘和细节
   方法:
   - USM: 最常用
   - Laplacian: 细节增强
   参数: 强度、半径
            ''',
            'CCM': '''
🎨 CCM (色彩校正矩阵)
   作用: 校准传感器色彩响应
   说明: 将传感器RGB转换到标准色彩空间
   参数: 3x3矩阵系数
            ''',
            'Gamma': '''
📈 Gamma (Gamma校正)
   作用: 亮度非线性校正
   常用值: 2.2 (sRGB), 2.4 (Rec.709)
   作用: 人眼对亮度敏感度是非线性的
            ''',
            'LTM': '''
🌗 LTM (局部色调映射)
   作用: 压缩高动态范围
   效果: 保留高光和暗部细节
   参数: 对比度强度、局部窗口大小
            ''',
            'CSC': '''
🔃 CSC (色彩空间转换)
   作用: 转换到输出色彩空间
   常用: sRGB, Adobe RGB, Rec.2020
            ''',
        }
        return explanations.get(stage_name, f"未知模块: {stage_name}")
    
    def generate_config(
        self,
        scene_type: str,
        quality: str = 'high'
    ) -> Dict[str, Any]:
        """生成场景推荐的Pipeline配置"""
        configs = {
            'landscape': {
                'Denoise': {'strength': 0.7, 'method': 'bilateral'},
                'Sharpening': {'strength': 1.2, 'radius': 1.5},
                'Saturation': {'value': 1.2},
                'LTM': {'enabled': True, 'strength': 0.6},
            },
            'portrait': {
                'Denoise': {'strength': 1.2, 'method': 'nlm'},
                'Sharpening': {'strength': 0.8, 'radius': 0.8},
                'Saturation': {'value': 0.9},
                'LTM': {'enabled': False},
            },
            'night': {
                'Denoise': {'strength': 1.5, 'method': 'nlm'},
                'Sharpening': {'strength': 1.1, 'radius': 1.0},
                'AWB': {'mode': 'gray_world'},
                'LTM': {'enabled': True, 'strength': 0.8},
            },
            'automotive': {
                'Denoise': {'strength': 1.0, 'method': 'bilateral'},
                'Sharpening': {'strength': 1.3, 'radius': 1.2},
                'Gamma': {'value': 2.2},
                'Contrast': {'value': 1.15},
                'LTM': {'enabled': True, 'strength': 0.5},
            },
        }
        
        base = {stage.name: {'enabled': stage.enabled} for stage in self.STANDARD_PIPELINE}
        
        if scene_type in configs:
            for module, params in configs[scene_type].items():
                if module in base:
                    base[module].update(params)
        
        return base


def create_pipeline_visualizer() -> ISPPipelineVisualizer:
    """创建Pipeline可视化器"""
    return ISPPipelineVisualizer()
