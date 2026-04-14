#!/usr/bin/env python3
"""
RAW格式处理模块
支持DNG/CR2/NEF/ARW等RAW格式

Phase 2.2 新增:
- HDR曝光合成 (Exposure Fusion)
- 多帧图像对齐
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from ..utils.logger import setup_logger

logger = setup_logger('isp-agent.raw')


class RawProcessor:
    """
    RAW格式处理器
    
    支持格式:
    - DNG (Adobe Digital Negative)
    - CR2 (Canon)
    - NEF (Nikon)
    - ARW (Sony)
    - RAF (Fujifilm)
    - RW2 (Panasonic)
    """
    
    SUPPORTED_FORMATS = ['.dng', '.cr2', '.nef', '.arw', '.raf', '.rw2', '.orf']
    
    def __init__(self):
        self._check_rawpy()
    
    def _check_rawpy(self):
        """检查rawpy是否可用"""
        try:
            import rawpy
            self.rawpy = rawpy
        except ImportError:
            logger.warning("rawpy未安装，RAW处理功能受限")
            self.rawpy = None
    
    def get_info(self, raw_path: str) -> Dict[str, Any]:
        """
        获取RAW文件信息
        
        Args:
            raw_path: RAW文件路径
        
        Returns:
            Dict: RAW文件信息
        """
        logger.info(f"读取RAW信息: {raw_path}")
        
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {raw_path}")
        
        info = {
            'file_name': path.name,
            'file_size': path.stat().st_size,
            'format': path.suffix.upper(),
        }
        
        if self.rawpy:
            try:
                with self.rawpy.imread(raw_path) as raw:
                    info.update({
                        'width': raw.raw_dimensions[0],
                        'height': raw.raw_dimensions[1],
                        'pattern': str(raw.color_pattern),  # Bayer pattern
                        'white_level': raw.white_level,
                        'black_level': raw.black_level_per_channel,
                        'iso': raw.iso_speed,
                        'exposure_time': raw.exposure_time,
                        'firmware': raw.firmware,
                        'timestamp': raw.timestamp if hasattr(raw, 'timestamp') else None,
                    })
                    
                    # 色彩信息
                    if hasattr(raw, 'color_desc'):
                        info['color_desc'] = raw.color_desc
                    
                    # 镜头信息
                    if hasattr(raw, 'lens'):
                        info['lens'] = str(raw.lens)
                    
                    logger.info(f"RAW信息读取成功: {info['width']}x{info['height']}")
                    
            except Exception as e:
                logger.error(f"读取RAW信息失败: {e}")
                info['error'] = str(e)
        else:
            info['warning'] = "rawpy未安装，无法读取详细信息"
        
        return info
    
    def process(
        self,
        raw_path: str,
        output_path: str,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        处理RAW文件
        
        Args:
            raw_path: RAW文件路径
            output_path: 输出路径
            settings: 处理设置
        
        Returns:
            Dict: 处理结果
        """
        logger.info(f"处理RAW文件: {raw_path}")
        
        settings = settings or {}
        
        if not self.rawpy:
            return {'success': False, 'error': 'rawpy未安装'}
        
        try:
            with self.rawpy.imread(raw_path) as raw:
                # 处理参数
                demosaic = settings.get('demosaic', True)
                use_camera_wb = settings.get('use_camera_wb', True)
                use_camera_nr = settings.get('use_camera_nr', False)
                no_auto_bright = settings.get('no_auto_bright', False)
                output_bps = settings.get('output_bps', 8)
                
                # Post processing设置
                pp = self.rawpy.PostProcessor()
                
                if demosaic:
                    pass  # 默认开启
                
                # 应用设置
                if 'brightness' in settings:
                    pp.brightness(settings['brightness'])
                
                if 'gamma' in settings:
                    pp.gamma(settings['gamma'])
                
                if 'no_auto_bright' in settings:
                    pp.no_auto_bright()
                
                # 执行处理
                rgb = raw.postprocess(
                    use_camera_wb=use_camera_wb,
                    use_camera_nr=use_camera_nr,
                    no_auto_bright=no_auto_bright,
                    output_bps=output_bps,
                )
                
                # 保存
                from PIL import Image
                img = Image.fromarray(rgb)
                img.save(output_path)
                
                logger.info(f"RAW处理完成: {output_path}")
                
                return {
                    'success': True,
                    'output': output_path,
                    'width': rgb.shape[1],
                    'height': rgb.shape[0],
                }
                
        except Exception as e:
            logger.error(f"RAW处理失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def extract_raw_data(self, raw_path: str) -> Optional[Dict[str, Any]]:
        """
        提取RAW原始数据用于分析
        
        Args:
            raw_path: RAW文件路径
        
        Returns:
            Dict: 原始数据信息
        """
        if not self.rawpy:
            return None
        
        try:
            with self.rawpy.imread(raw_path) as raw:
                # 获取原始数据
                raw_data = raw.raw_image_visible
                
                # 统计信息
                stats = {
                    'min': int(raw_data.min()),
                    'max': int(raw_data.max()),
                    'mean': float(raw_data.mean()),
                    'std': float(raw_data.std()),
                }
                
                # Bayer pattern分析
                pattern = str(raw.color_pattern)
                
                return {
                    'dimensions': raw.raw_dimensions,
                    'pattern': pattern,
                    'stats': stats,
                    'black_level': raw.black_level_per_channel,
                    'white_level': raw.white_level,
                }
                
        except Exception as e:
            logger.error(f"提取RAW数据失败: {e}")
            return None
    
    def to_tiff(self, raw_path: str, output_path: str) -> Dict[str, Any]:
        """转换为TIFF格式"""
        return self.process(raw_path, output_path, {'output_bps': 16})
    
    def to_jpeg(self, raw_path: str, output_path: str, quality: int = 95) -> Dict[str, Any]:
        """转换为JPEG格式"""
        return self.process(
            raw_path, 
            output_path, 
            {'output_bps': 8}
        )


def process_raw(raw_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
    """便捷函数"""
    processor = RawProcessor()
    return processor.process(raw_path, output_path, kwargs)


def get_raw_info(raw_path: str) -> Dict[str, Any]:
    """便捷函数"""
    processor = RawProcessor()
    return processor.get_info(raw_path)


# =============================================================================
# Phase 2.2: RAW HDR合成 (Exposure Fusion)
# =============================================================================

def synthesize_hdr_exposures(
    images: List[np.ndarray],
    exposure_times: List[float]
) -> np.ndarray:
    """
    多帧曝光融合HDR合成 (Exposure Fusion)

    无需HDR校准的多帧融合算法，适用于手持拍摄或无精确曝光时间的场景。
    算法基于Mertens et al. "Exposure Fusion" (2007)的简化实现。

    曝光质量权重计算:
    - 对比度 (Contrast): 使用拉普拉斯算子检测边缘/细节区域
    - 饱和度 (Saturation): 高饱和度区域通常是正确曝光的区域
    - 良好曝光 (Well-exposedness): 像素值接近中灰 (0.5) 优先

    最终权重 = Contrast^w1 * Saturation^w2 * Exposure^w3

    Args:
        images: 多帧不同曝光的图像列表，每帧为 uint8 [0,255] 或 float [0,1]
        exposure_times: 对应每帧的曝光时间列表 (相对值，如 [1/125, 1/60, 1/30])

    Returns:
        HDR图像: 32位float，线性空间，范围 [0, 1] (做了归一化)

    Reference:
        Mertens et al., "Exposure Fusion", Pacific Graphics 2007
    """
    if len(images) < 2:
        raise ValueError("至少需要2帧图像进行HDR合成")
    if len(images) != len(exposure_times):
        raise ValueError("图像数量与曝光时间数量不匹配")

    logger.info(f"开始HDR曝光融合: {len(images)} 帧")

    # ---------- 预处理: 统一格式 ----------
    imgs_float = []
    for img in images:
        if img.dtype == np.uint8:
            imgs_float.append(img.astype(np.float32) / 255.0)
        else:
            imgs_float.append(img.astype(np.float32))

    h, w = imgs_float[0].shape[:2]
    n_frames = len(imgs_float)

    # ---------- 计算每帧的曝光质量权重 ----------
    # 权重图: 每帧每个像素的融合权重
    weight_maps = []

    for i, img in enumerate(imgs_float):
        # 灰度图 (用于权重计算)
        if len(img.shape) == 3:
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.7142 * img[:, :, 2]
        else:
            gray = img.copy()

        # 1) 对比度权重 (Laplacian)
        # 使用Sobel近似拉普拉斯，检测细节丰富区域
        laplacian = _laplacian_contrast(gray)

        # 2) 饱和度权重 (标准差)
        if len(img.shape) == 3:
            # 每像素的饱和度用通道标准差近似
            sat = _saturation_measure(img)
        else:
            sat = np.zeros_like(gray)

        # 3) 良好曝光权重 (偏离中灰的程度，越接近中灰越好)
        # exposure_score = exp(-12 * (gray - 0.5)^2)，值域 [0,1]
        exposure_score = np.exp(-12.0 * (gray - 0.5) ** 2)

        # 合并权重 (使用幂次归一化避免某一项主导)
        # w1=w2=w3=1 (等权重)
        eps = 1e-8
        w_contrast = np.power(np.maximum(laplacian, eps), 1.0)
        w_sat = np.power(np.maximum(sat, eps), 1.0)
        w_exposure = np.power(np.maximum(exposure_score, eps), 1.0)

        # 最终权重 (逐像素归一化，避免融合时总权重不稳定)
        weight = w_contrast * w_sat * w_exposure

        # 如果曝光时间可用，用曝光时间做参考修正 (曝光越长越亮)
        # 不改变权重计算逻辑，只用于记录
        weight_maps.append(weight)

    # ---------- 逐像素归一化权重 ----------
    weight_stack = np.stack(weight_maps, axis=-1)  # (H, W, N)
    weight_sum = np.sum(weight_stack, axis=-1, keepdims=True)  # (H, W, 1)
    weight_sum = np.maximum(weight_sum, eps)
    weight_norm = weight_stack / weight_sum  # (H, W, N)

    # ---------- 多尺度融合 (Laplac Pyramid) ----------
    num_levels = 6  # 金字塔层数
    hdr_result = np.zeros((h, w, 3) if len(imgs_float[0].shape) == 3
                          else (h, w), dtype=np.float32)

    if len(imgs_float[0].shape) == 3:
        # 彩色图像: 分别处理每个通道
        for c in range(3):
            channel_pyramids = []
            for img in imgs_float:
                channel_pyramids.append(img[:, :, c])

            # 构建拉普拉斯金字塔
            lap_pyramids = [_build_laplacian_pyramid(ch, num_levels)
                           for ch in channel_pyramids]

            # 加权融合每层
            fused_pyramid = []
            for level in range(num_levels):
                level_channels = [p[level] for p in lap_pyramids]
                # 该层的权重
                w_level = _downsample_weight_map(weight_norm, level)
                # 融合
                if level == num_levels - 1:
                    # 顶层: w_level shape (H',W',N), squeeze到(H',W',N) 但ch是(H',W')
                    # 正确做法: w_level[:,:,i] -> (H',W') 与 ch (H',W') 相乘
                    fused_ch = sum(w_level[:, :, i] * ch
                                   for i, ch in enumerate(level_channels))
                else:
                    # 正常层: w_level[:,:,i] -> (H',W')
                    fused_ch = sum(w_level[:, :, i] * ch
                                   for i, ch in enumerate(level_channels))
                fused_pyramid.append(fused_ch)

            # 从金字塔重建
            hdr_result[:, :, c] = _collapse_laplacian_pyramid(fused_pyramid)
    else:
        # 灰度图像
        gray_pyramids = [_build_laplacian_pyramid(img, num_levels)
                        for img in imgs_float]
        fused_pyramid = []
        for level in range(num_levels):
            level_imgs = [p[level] for p in gray_pyramids]
            w_level = _downsample_weight_map(weight_norm, level)
            # 统一用 w_level[:, :, i]
            fused = sum(w_level[:, :, i] * ch
                        for i, ch in enumerate(level_imgs))
            fused_pyramid.append(fused)
        hdr_result = _collapse_laplacian_pyramid(fused_pyramid)

    # ---------- 后处理: 线性空间归一化 ----------
    # HDR值可能超出[0,1]，不做硬截断，保留相对关系
    hdr_min = np.min(hdr_result)
    hdr_max = np.max(hdr_result)
    if hdr_max > hdr_min + eps:
        hdr_result = (hdr_result - hdr_min) / (hdr_max - hdr_min)

    logger.info(f"HDR合成完成: 动态范围 {hdr_max - hdr_min:.2f} (归一化后 [0,1])")
    return hdr_result.astype(np.float32)


def align_exposures(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    多帧图像对齐 (基于特征点匹配)

    使用SIFT/ORB特征检测 + RANSAC单应性估计来对齐图像，
    处理手持拍摄导致的轻微抖动/偏移。

    Args:
        images: 未对齐的图像列表 (uint8或float, BGR或Gray)

    Returns:
        对齐后的图像列表 (与输入尺寸相同)

    Note:
        纯NumPy实现，但特征检测需要cv2(OpenCV)。
        若无OpenCV则返回原始图像并记录警告。
    """
    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False
        logger.warning("OpenCV未安装，align_exposures使用朴素对齐(仅裁边)")
        return _align_exposures_naive(images)

    if len(images) < 2:
        return images

    logger.info(f"开始图像对齐: {len(images)} 帧")

    # 预处理: 转uint8灰度
    def to_gray(img):
        if img.dtype != np.uint8:
            img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img_u8 = img
        if len(img_u8.shape) == 3:
            gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_u8
        return gray

    # 选择参考帧 (中间帧，避免极端曝光)
    ref_idx = len(images) // 2
    ref_gray = to_gray(images[ref_idx])
    h, w = ref_gray.shape

    aligned = [images[ref_idx].copy()]  # 参考帧不变

    for i, img in enumerate(images):
        if i == ref_idx:
            continue

        gray = to_gray(img)
        img_h, img_w = gray.shape

        # 缩放比例 (用于SIFT特征坐标还原)
        scale_x = w / img_w if img_w != w else 1.0
        scale_y = h / img_h if img_h != h else 1.0

        # 统一尺寸到参考帧大小
        if img_h != h or img_w != w:
            gray_scaled = cv2.resize(gray, (w, h))
        else:
            gray_scaled = gray

        # ---------- 特征点检测 (SIFT) ----------
        sift = cv2.SIFT_create(nfeatures=2000)
        kp1, des1 = sift.detectAndCompute(ref_gray, None)
        kp2, des2 = sift.detectAndCompute(gray_scaled, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            # 特征点不足，使用原始图像
            aligned.append(images[i].copy())
            continue

        # ---------- 特征匹配 (BFMatcher) ----------
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = []
        for m_n in matches:
            if len(m_n) == 2 and m_n[0].distance < 0.75 * m_n[1].distance:
                good.append(m_n[0])

        if len(good) < 10:
            aligned.append(images[i].copy())
            continue

        # ---------- 单应性矩阵估计 (RANSAC) ----------
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

        if M is None:
            aligned.append(images[i].copy())
            continue

        # 检查RANSAC内点率
        inliers = mask.ravel().sum()
        inlier_ratio = inliers / len(good)
        logger.debug(f"帧{i}: {len(good)} 匹配点, RANSAC内点率 {inlier_ratio:.2%}")

        if inlier_ratio < 0.1:
            # 内点率过低，跳过对齐
            aligned.append(images[i].copy())
            continue

        # ---------- 透视变换 ----------
        # 输出尺寸与参考帧相同
        img_u8 = (np.clip(images[i], 0, 1) * 255).astype(np.uint8) \
            if images[i].dtype != np.uint8 else images[i].copy()

        if len(img_u8.shape) == 3:
            aligned_img = cv2.warpPerspective(
                img_u8, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
        else:
            aligned_img = cv2.warpPerspective(
                img_u8, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )

        # 转回原始类型
        if images[i].dtype != np.uint8:
            aligned_img = aligned_img.astype(np.float32) / 255.0

        aligned.append(aligned_img)

    logger.info(f"图像对齐完成: {len(aligned)} 帧")
    return aligned


# =============================================================================
# 内部辅助函数 (高效实现)
# =============================================================================

def _laplacian_contrast(gray: np.ndarray) -> np.ndarray:
    """
    计算灰度图的拉普拉斯对比度 (向量化, 无循环)

    Args:
        gray: 灰度图 float [0,1]

    Returns:
        对比度图 (正值越大=对比度越高)
    """
    g = gray.astype(np.float64)
    # Sobel算子使用 np.roll 实现 (无循环)
    # gx ≈ (g[i,j+1] - g[i,j-1]) / 2
    gx = (np.roll(g, -1, axis=1) - np.roll(g, 1, axis=1)) / 2.0
    gy = (np.roll(g, -1, axis=0) - np.roll(g, 1, axis=0)) / 2.0
    # 边缘填充回原值
    gx[:, 0] = g[:, 1] - g[:, 0]
    gx[:, -1] = g[:, -1] - g[:, -2]
    gy[0, :] = g[1, :] - g[0, :]
    gy[-1, :] = g[-1, :] - g[-2, :]

    contrast = np.sqrt(gx ** 2 + gy ** 2)
    c_max = np.max(contrast) + 1e-8
    return (contrast / c_max).astype(np.float32)


def _saturation_measure(img: np.ndarray) -> np.ndarray:
    """
    计算彩色图像的饱和度 (通道标准差，向量化)

    Args:
        img: float [0,1], shape (H,W,3)

    Returns:
        饱和度图 (H,W)
    """
    eps = 1e-8
    mean = np.mean(img, axis=-1, keepdims=True)
    var = np.mean((img - mean) ** 2, axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    s_max = np.max(std) + eps
    return np.squeeze(std / s_max).astype(np.float32)


def _build_laplacian_pyramid(img: np.ndarray, num_levels: int) -> List[np.ndarray]:
    """
    构建拉普拉斯金字塔

    Args:
        img: 2D图像 float
        num_levels: 金字塔层数

    Returns:
        金字塔列表 (从底层到顶层)
    """
    try:
        from scipy import ndimage
        has_scipy = True
    except ImportError:
        has_scipy = False

    pyramid = []
    current = img.copy()

    for _ in range(num_levels - 1):
        # 高斯模糊 + 下采样
        if has_scipy:
            blurred = ndimage.gaussian_filter(current, sigma=2.0)
        else:
            blurred = _gaussian_blur_2d(current, sigma=2.0)

        h, w = blurred.shape
        if h < 8 or w < 8:
            break
        downsampled = _downsample_2x(blurred)
        # 上采样回原尺寸
        upsampled = _upsample_2x(downsampled, target_h=h, target_w=w)
        # 拉普拉斯层 = 原始 - 预测
        lap = current - upsampled
        pyramid.append(lap)
        current = downsampled

    pyramid.append(current)  # 最顶层
    return pyramid


def _collapse_laplacian_pyramid(pyramid: List[np.ndarray]) -> np.ndarray:
    """从拉普拉斯金字塔重建图像"""
    current = pyramid[-1]
    for level in reversed(pyramid[:-1]):
        # 上采样
        upsampled = _upsample_2x(current, target_h=level.shape[0], target_w=level.shape[1])
        current = upsampled + level
    return np.clip(current, 0, 1)


def _gaussian_blur_2d(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    2D高斯模糊

    优先使用cv2 (OpenCV)，回退到scipy，最后用纯numpy einsum实现。
    """
    # 尝试cv2 (最快)
    try:
        import cv2
        ksize = max(int(6 * sigma + 1), 3)
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(img.astype(np.float32), (ksize, ksize), sigma, sigma)
    except Exception:
        pass

    # 尝试scipy
    try:
        from scipy import ndimage
        return ndimage.gaussian_filter(img, sigma=sigma).astype(np.float32)
    except Exception:
        pass

    # 纯numpy einsum分离卷积 (高效，无Python循环)
    return _gaussian_blur_einsum(img, sigma)


def _gaussian_blur_einsum(img: np.ndarray, sigma: float) -> np.ndarray:
    """纯numpy高斯模糊: einsum滑动窗口实现"""
    radius = max(int(3 * sigma + 0.5), 1)
    ksize = 2 * radius + 1

    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    img_f = img.astype(np.float32)
    h, w = img_f.shape

    # 水平卷积: (H, W) → (H, W-K+1)
    def h_conv(arr: np.ndarray) -> np.ndarray:
        shape = (arr.shape[0], arr.shape[1] - ksize + 1)
        strides = (arr.strides[0], arr.strides[1])
        patches = np.lib.stride_tricks.as_strides(
            arr, shape=shape, strides=strides, writeable=False
        )
        return np.einsum('hwk,k->hw', patches, kernel, optimize=True)

    # 垂直卷积: 用 h_conv 对转置后的数组操作
    blurred_h = h_conv(img_f)
    # 垂直需要对称填充后处理
    blurred_h_pad = np.pad(blurred_h, ((radius, radius), (0, 0)), mode='symmetric')
    blurred = h_conv(blurred_h_pad.T).T
    return blurred[:h, :w]


def _downsample_2x(img: np.ndarray) -> np.ndarray:
    """2x下采样"""
    return img[::2, ::2]


def _upsample_2x(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """上采样到目标尺寸 (最近邻 + 高斯平滑)"""
    h, w = img.shape
    upsampled = np.zeros((h * 2, w * 2), dtype=img.dtype)
    upsampled[::2, ::2] = img
    upsampled[1::2, ::2] = img
    upsampled[::2, 1::2] = img
    upsampled[1::2, 1::2] = img
    # 高斯模糊去锯齿
    blurred = _gaussian_blur_2d(upsampled, sigma=0.8)
    return blurred[:target_h, :target_w]


def _downsample_weight_map(weight_norm: np.ndarray, level: int) -> np.ndarray:
    """将权重图下采样指定层级"""
    w = weight_norm.copy()
    for _ in range(level):
        h, wi, n = w.shape
        if h < 4 or wi < 4:
            break
        w = w[::2, ::2, :]
    return w


def _align_exposures_naive(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    朴素对齐 (无OpenCV): 中心裁剪 + 轻微平移
    """
    if not images:
        return []

    # 以第一帧为参考
    ref = images[0]
    h, w = ref.shape[:2]

    # 裁剪边距 (假设偏移不超过5%)
    margin_h = int(h * 0.05)
    margin_w = int(w * 0.05)

    # 中心裁剪参考帧
    ref_crop = ref[margin_h: h - margin_h, margin_w: w - margin_w]

    aligned = [ref_crop]
    for img in images[1:]:
        ih, iw = img.shape[:2]
        mh = int(ih * 0.05)
        mw = int(iw * 0.05)
        img_crop = img[mh: ih - mh, mw: iw - mw]
        # 缩放到相同尺寸
        if img_crop.shape[:2] != ref_crop.shape[:2]:
            import cv2
            img_crop = cv2.resize(img_crop, (ref_crop.shape[1], ref_crop.shape[0]))
        aligned.append(img_crop)

    return aligned


# =============================================================================
# 便捷函数
# =============================================================================

def hdr_synthesize(
    images: List[np.ndarray],
    exposure_times: List[float]
) -> np.ndarray:
    """
    多帧HDR曝光融合便捷函数

    Args:
        images: 图像列表 (uint8或float)
        exposure_times: 曝光时间列表

    Returns:
        HDR图像 (float32, [0,1])
    """
    return synthesize_hdr_exposures(images, exposure_times)


def align_images(
    images: List[np.ndarray]
) -> List[np.ndarray]:
    """
    多帧图像对齐便捷函数

    Args:
        images: 图像列表

    Returns:
        对齐后的图像列表
    """
    return align_exposures(images)

