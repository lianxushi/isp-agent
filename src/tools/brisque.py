#!/usr/bin/env python3
"""
BRISQUE - Blind/Referenceless Image Spatial Quality Evaluator
基于自然场景统计(NSS)的无参考图像质量评估

BRISQUE使用自然图像的统计特性来预测感知质量，无需参考图像。
核心思想：自然图像经过局部归一化对比度(LNC)处理后，呈现近似高斯分布。
图像失真会改变这种统计特性，从而可以用于质量评估。

Reference: "No-Reference Image Quality Assessment in the Spatial Domain"
            Mittal et al., IEEE TPAMI 2012
"""
import cv2
import numpy as np
import math
from typing import Dict, Any, Tuple, Optional, List
from ..utils.logger import setup_logger

logger = setup_logger('isp-agent.brisque')


# ============================================================
# 纯NumPy实现，不依赖scipy
# ============================================================

def _gamma(x: float) -> float:
    """
    Gamma函数 - 使用Python标准库math.gamma
    替代scipy.special.gamma，纯标准库实现，无需scipy依赖。
    """
    if x <= 0:
        return float('inf')
    try:
        return math.gamma(x)
    except (ValueError, OverflowError):
        return float('inf')


def _kurtosis(data: np.ndarray, fisher: bool = True) -> float:
    """
    计算峰度 (Kurtosis) - 纯NumPy实现，替代scipy.stats.kurtosis

    Args:
        data: 输入数组
        fisher: True=Fisher定义 (正态=0), False=Pearson定义 (正态=3)

    Returns:
        峰度值 (Fisher: 正态分布=0)
    """
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]

    if len(data) < 4:
        return 0.0

    mean = np.mean(data)
    std = np.std(data, ddof=0)

    if std < 1e-10:
        return 0.0

    # 4阶中心矩 / 标准差的4次方 - 3 (Fisher)
    m4 = np.mean((data - mean) ** 4)
    k = m4 / (std ** 4)

    if fisher:
        k -= 3.0  # Fisher: 正态分布 = 0

    return float(k)


def _estimate_ggd_param(block: np.ndarray) -> Tuple[float, float]:
    """
    估计广义高斯分布(Generalized Gaussian Distribution)参数

    GGD pdf: p(x; α, β) = (β / (2α * Γ(1/β))) * exp(-(|x|/α)^β)

    其中:
    - α: 尺度参数 (scale)
    - β: 形状参数 (shape), β=2 时为高斯分布

    Args:
        block: 输入数据 (1D array)

    Returns:
        Tuple[α, β]: 尺度参数和形状参数
    """
    block = block.flatten()
    block = block[~np.isnan(block)]

    if len(block) < 10:
        return 1.0, 2.0

    # 计算矩
    r_hat = np.mean(np.abs(block) ** 2)
    r_hat_abs = np.mean(np.abs(block))

    if r_hat_abs < 1e-10:
        return 1.0, 2.0

    # 形状参数估计 - 使用峰度来估计形状参数
    k = _kurtosis(block, fisher=True)
    if k > 0:
        # 正峰度 -> 重尾分布 -> β < 2
        beta = max(0.1, min(10.0, 2.0 / (np.log(k + 1) + 1)))
    else:
        # 负峰度 -> 均匀分布 -> β > 2
        beta = 2.0

    # 尺度参数估计
    if abs(beta - 2.0) < 0.01:
        alpha = np.sqrt(r_hat)
    else:
        try:
            # alpha = sqrt(r_hat * Γ(1/β) / Γ(3/β))
            alpha = np.sqrt(r_hat * _gamma(1.0/beta) / _gamma(3.0/beta))
        except (ValueError, ZeroDivisionError):
            alpha = np.sqrt(r_hat)

    if alpha < 1e-10:
        alpha = 1.0

    return float(alpha), float(beta)


def _estimate_agd_param(x: np.ndarray) -> Tuple[float, float, float, float]:
    """
    估计非对称高斯分布(Asymmetric Gaussian Distribution)参数
    用于拟合MSC系数邻居乘积

    AGD pdf: p(x; α_l, α_r, μ) =
        (1/(α_l * Γ(1/μ))) * exp(-(-x/α_l)^μ)  for x < 0
        (1/(α_r * Γ(1/μ))) * exp(-(x/α_r)^μ)    for x >= 0

    Args:
        x: 输入数据 (1D array)

    Returns:
        Tuple[α_l, α_r, μ, skew]: 左尺度、右尺度、形状参数、偏度
    """
    x = x.flatten()
    x = x[~np.isnan(x)]

    if len(x) < 10:
        return 1.0, 1.0, 1.0, 0.0

    left = x[x < 0]
    right = x[x >= 0]

    # 参数估计
    alpha_l = np.sqrt(np.mean(left ** 2)) if len(left) > 0 else 1.0
    alpha_r = np.sqrt(np.mean(right ** 2)) if len(right) > 0 else 1.0

    if alpha_l < 1e-10:
        alpha_l = 1.0
    if alpha_r < 1e-10:
        alpha_r = 1.0

    mu = 1.0  # 简化假设

    # 偏度和峰度
    std = np.std(x)
    if std < 1e-10:
        skew = 0.0
    else:
        skew = np.mean(((x - np.mean(x)) / std) ** 3)

    return float(alpha_l), float(alpha_r), float(mu), float(skew)


class BRISQUE:
    """
    BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)

    实现步骤:
    1. 计算局部归一化对比度(MSCN系数)
    2. 计算邻居系数乘积(水平、垂直、对角方向)
    3. 拟合广义高斯分布(GGD)获取形状参数
    4. 拟合非对称高斯分布(AGD)获取参数
    5. 组成特征向量，使用SVR回归预测MOS分数

    特征维度:
    - MSCN系数: 2个参数 (α, β)
    - 水平乘积: 3个参数 (α_l, α_r, skew)
    - 垂直乘积: 3个参数 (α_l, α_r, skew)
    - 对角1乘积: 3个参数 (α_l, α_r, skew)
    - 对角2乘积: 3个参数 (α_l, α_r, skew)
    - 共: 2 + 4*3 = 14个特征 (单尺度)
    - 2尺度: 28个特征
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        初始化BRISQUE

        Args:
            model_path: 预训练的SVR模型路径(可选)
        """
        self.model = None
        self.model_path = model_path
        self._load_model()

        # 预训练的BRISQUE参数 (用于零样本评估)
        self._default_mean = 0.0
        self._default_std = 1.0

        # 质量等级阈值 (基于大量实验统计)
        self._quality_thresholds = {
            'excellent': (80, 100),
            'good': (60, 80),
            'fair': (40, 60),
            'poor': (20, 40),
            'bad': (0, 20)
        }

    def _load_model(self) -> None:
        """加载预训练模型"""
        if self.model_path:
            try:
                import pickle
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"已加载BRISQUE模型: {self.model_path}")
            except Exception as e:
                logger.warning(f"无法加载模型: {e}，将使用简化评分")
                self.model = None

    def _compute_mscn(self, img: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        """
        计算Mean Subtracted Contrast Normalized (MSCN)系数

        MSCN = (I - μ) / σ

        其中μ和σ是局部均值和标准差

        Args:
            img: 输入灰度图像
            kernel_size: 局部窗口大小

        Returns:
            MSCN系数矩阵
        """
        C = 1.0 / 255  # 防止除零

        # 计算局部均值
        mu = cv2.blur(img.astype(np.float32), (kernel_size, kernel_size))

        # 计算局部方差
        mu_sq = mu ** 2
        sigma = cv2.blur(img.astype(np.float32) ** 2, (kernel_size, kernel_size)) - mu_sq
        sigma = np.sqrt(np.maximum(sigma, 0)) + C

        # 计算MSCN
        mscn = (img.astype(np.float32) - mu) / sigma

        return mscn.astype(np.float32)

    def _compute_pairwise_products(self, mscn: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算邻居系数乘积

        方向: 水平(H)、垂直(V)、主对角(D1)、副对角(D2)

        Args:
            mscn: MSCN系数矩阵

        Returns:
            各方向的乘积字典
        """
        h, w = mscn.shape

        # 水平方向 (H)
        H = mscn[:, :-1] * mscn[:, 1:]

        # 垂直方向 (V)
        V = mscn[:-1, :] * mscn[1:, :]

        # 主对角方向 (D1) - 右下对角
        D1 = mscn[:-1, :-1] * mscn[1:, 1:]

        # 副对角方向 (D2) - 左下对角
        D2 = mscn[:-1, 1:] * mscn[1:, :-1]

        return {
            'H': H,
            'V': V,
            'D1': D1,
            'D2': D2
        }

    def _extract_features_single_scale(
        self,
        img: np.ndarray,
        kernel_size: int = 7
    ) -> List[float]:
        """
        单尺度BRISQUE特征提取

        Args:
            img: 灰度图像
            kernel_size: 窗口大小

        Returns:
            特征向量 (14维)
        """
        features = []

        # 计算MSCN
        mscn = self._compute_mscn(img, kernel_size)

        # MSCN系数的GGD参数
        alpha, beta = _estimate_ggd_param(mscn)
        features.extend([alpha, beta])

        # 邻居系数乘积
        products = self._compute_pairwise_products(mscn)

        for direction in ['H', 'V', 'D1', 'D2']:
            product = products[direction]
            alpha_l, alpha_r, mu, skew = _estimate_agd_param(product)
            # AGD参数: 左尺度、右尺度、形状、偏度
            features.extend([alpha_l, alpha_r, skew])

        return features

    def _extract_features_multi_scale(
        self,
        img: np.ndarray,
        kernel_sizes: List[int] = [7, 15]
    ) -> List[float]:
        """
        多尺度BRISQUE特征提取

        Args:
            img: 灰度图像
            kernel_sizes: 不同尺度的窗口大小列表

        Returns:
            特征向量 (14 * len(kernel_sizes) 维)
        """
        all_features = []

        for kernel_size in kernel_sizes:
            features = self._extract_features_single_scale(img, kernel_size)
            all_features.extend(features)

        return all_features

    def extract_features(self, img: np.ndarray) -> Dict[str, Any]:
        """
        提取BRISQUE特征

        Args:
            img: 彩色或灰度图像

        Returns:
            特征字典
        """
        # 转为灰度
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 转换为float
        gray = gray.astype(np.float32)

        # 多尺度特征
        features_7 = self._extract_features_single_scale(gray, 7)
        features_15 = self._extract_features_single_scale(gray, 15)

        all_features = features_7 + features_15

        mscn = self._compute_mscn(gray)

        return {
            'features': all_features,
            'num_features': len(all_features),
            'scale_7_features': features_7,
            'scale_15_features': features_15,
            'mscn_mean': float(np.mean(mscn)),
            'mscn_std': float(np.std(mscn)),
            'mscn_kurtosis': float(_kurtosis(mscn))
        }

    def _normalized_distance(self, features: np.ndarray) -> float:
        """
        计算特征到"高质量图像特征分布"的归一化距离

        高质量自然图像的MSCN系数近似标准正态分布，
        偏离这个分布越远，质量越差

        Args:
            features: BRISQUE特征向量

        Returns:
            质量分数 (0-100, 100=最佳)
        """
        if len(features) >= 2:
            alpha, beta = features[0], features[1]

            # 理想情况下 alpha ≈ 1.0 (标准差), beta ≈ 2.0 (高斯)
            # 偏离越大，分数越低

            # 形状参数偏差
            shape_score = max(0, 100 - abs(beta - 2.0) * 20)

            # 尺度参数偏差
            scale_score = max(0, 100 - abs(alpha - 1.0) * 30)

            # 基础分数
            base_score = (shape_score * 0.4 + scale_score * 0.6)
        else:
            base_score = 50.0

        return min(100, max(0, base_score))

    def _compute_quality_from_features(self, features: List[float]) -> Dict[str, Any]:
        """
        从特征向量计算质量分数

        使用简化的统计模型替代完整的SVR回归
        基于BRISQUE特征的物理意义进行评分

        Args:
            features: BRISQUE特征向量

        Returns:
            质量评估结果
        """
        if self.model is not None:
            # 使用预训练模型
            try:
                score = self.model.predict([features])[0]
                score = max(0, min(100, score))
            except:
                score = self._normalized_distance(np.array(features))
        else:
            # 使用简化评分
            score = self._normalized_distance(np.array(features))

        # 转换为1-5的MOS分数
        mos = 1.0 + 4.0 * (score / 100.0)

        return {
            'brisque_score': round(score, 2),  # 0-100
            'mos_predicted': round(mos, 2),    # 1-5
            'quality_level': self._get_quality_level(score)
        }

    def _get_quality_level(self, score: float) -> str:
        """根据分数确定质量等级"""
        if score >= 80:
            return 'excellent'
        elif score >= 60:
            return 'good'
        elif score >= 40:
            return 'fair'
        elif score >= 20:
            return 'poor'
        else:
            return 'bad'

    def assess(self, img: np.ndarray) -> Dict[str, Any]:
        """
        评估图像质量

        Args:
            img: 输入图像 (彩色或灰度)

        Returns:
            质量评估结果字典
        """
        # 提取特征
        feature_result = self.extract_features(img)

        # 计算质量分数
        quality_result = self._compute_quality_from_features(feature_result['features'])

        # 额外分析
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # MSCN分析
        mscn = self._compute_mscn(gray.astype(np.float32))
        mscn_analysis = self._analyze_mscn(mscn)

        return {
            'brisque_score': quality_result['brisque_score'],
            'mos_predicted': quality_result['mos_predicted'],
            'quality_level': quality_result['quality_level'],
            'num_features': feature_result['num_features'],
            'mscn_analysis': mscn_analysis,
            'naturalness': self._assess_naturalness(mscn),
            'details': {
                'alpha': feature_result['features'][0] if len(feature_result['features']) > 0 else 0,
                'beta': feature_result['features'][1] if len(feature_result['features']) > 1 else 0,
                'mscn_mean': feature_result['mscn_mean'],
                'mscn_std': feature_result['mscn_std'],
                'mscn_kurtosis': feature_result['mscn_kurtosis']
            }
        }

    def _analyze_mscn(self, mscn: np.ndarray) -> Dict[str, Any]:
        """分析MSCN系数特性"""
        kurt = _kurtosis(mscn, fisher=True)
        std = float(np.std(mscn))
        mean = float(np.mean(mscn))
        if std < 1e-10:
            skew = 0.0
        else:
            skew = float(np.mean(((mscn - mean) / std) ** 3))

        return {
            'mean': mean,
            'std': std,
            'kurtosis': kurt,
            'skewness': skew,
            'is_gaussian_like': bool(np.abs(kurt) < 1.0),
            'note': '自然图像的MSCN系数应接近标准正态分布(kurtosis≈0)'
        }

    def _assess_naturalness(self, mscn: np.ndarray) -> Dict[str, Any]:
        """
        评估图像"自然度"

        自然图像经过局部归一化后，其统计特性应接近高斯分布。
        人工处理过的图像(如过度压缩、滤波等)会偏离这个分布。

        Args:
            mscn: MSCN系数

        Returns:
            自然度评估
        """
        kurt = _kurtosis(mscn, fisher=True)
        std = float(np.std(mscn))
        if std < 1e-10:
            skew = 0.0
        else:
            skew = float(np.mean(((mscn - np.mean(mscn)) / std) ** 3))

        # 判断是否接近高斯 (峰度≈0, 偏度≈0)
        kurt_deviation = abs(kurt)
        skew_deviation = abs(skew)

        # 计算自然度分数
        kurt_score = max(0, 100 - kurt_deviation * 10)
        skew_score = max(0, 100 - skew_deviation * 50)
        naturalness = (kurt_score * 0.6 + skew_score * 0.4)

        return {
            'naturalness_score': round(naturalness, 2),
            'kurtosis_deviation': round(kurt_deviation, 3),
            'skewness_deviation': round(skew_deviation, 3),
            'is_natural': bool(naturalness > 70),
            'interpretation': self._interpret_naturalness(naturalness)
        }

    def _interpret_naturalness(self, score: float) -> str:
        """解释自然度分数"""
        if score >= 90:
            return "高度自然，统计特性接近原始自然图像"
        elif score >= 70:
            return "自然度良好，轻微处理痕迹"
        elif score >= 50:
            return "自然度一般，可能经过明显处理"
        elif score >= 30:
            return "自然度较差，强处理或压缩"
        else:
            return "严重失真，可能为合成图像或重度处理"


def compute_brisque_features(img: np.ndarray) -> List[float]:
    """便捷函数：计算BRISQUE特征"""
    brisque = BRISQUE()
    result = brisque.extract_features(img)
    return result['features']


def assess_quality_brisque(img: np.ndarray) -> Dict[str, Any]:
    """便捷函数：BRISQUE质量评估"""
    brisque = BRISQUE()
    return brisque.assess(img)
