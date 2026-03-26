# ISP-Agent v0.4 开发计划

## 版本目标
打造专业级ISP图像质量分析工具，重点提升：
1. 无参考质量评估精度
2. HDR处理能力
3. 车载场景分析深度
4. 代码质量与测试覆盖

## 交付时间
目标：2周内完成核心改进

---

## Phase 1: AI质量评估增强 (BRISQUE-style)

### 目标
实现基于自然场景统计(NSS)的无参考图像质量评估

### 实现内容
- [ ] `src/tools/brisque.py` - BRISQUE特征提取
  - MSCN (Mean Subtracted Contrast Normalized) 系数
  - 成对.products (邻居系数乘积)
  - 广义高斯分布拟合
  - 形状/尺度参数
  
- [ ] `src/tools/niqe.py` - NIQE质量评估
  - 自然场景统计模型
  - 多尺度特征提取
  - MVG (Multivariate Gaussian) 拟合

- [ ] 集成到 `AIQualityScorer`
  - 原有启发式评分 + BRISQUE分数融合
  - 提供更准确的MOS预测

### 技术方案
```
BRISQUE特征:
1. 对数导数(DivNorm/MSCN)
2. 4个方向的成对乘积 (水平、垂直、对角)
3. 广义高斯分布拟合 (α, σ²)
4. 2-scale / 1-scale 提取
5. 共计36个特征
6. SVR回归预测质量分数
```

---

## Phase 2: HDR增强

### 目标
提供专业级HDR处理能力

### 实现内容
- [ ] 多种Tone Mapping算法
  - [ ] Reinhard (全局+局部)
  - [ ] Mantiuk (对比度保留)
  - [ ] Drago (自适应对数)
  - [ ] ACES (学院色彩编码系统)
  
- [ ] RAW HDR合成
  - 多帧曝光融合
  - 包围曝光对齐
  - 动态范围扩展

- [ ] HDR质量评估
  - 有效动态范围计算
  - 高光/暗部保留评估
  - Tone mapping曲线分析

### 技术方案
```python
# Tone Mapping算法选择
TONEMAP_REINHARD = 'reinhard'     # 适合一般用途
TONEMAP_MANTIK = 'mantik'         # 保留对比度
TONEMAP_DRAGO = 'drago'           # 极端光照
TONEMAP_ACES = 'aces'            # 电影级色彩
```

---

## Phase 3: 车载场景深度分析

### 目标
对标ISO 16505汽车摄像头标准

### 实现内容
- [ ] ISO 16505合规检查
  - 分辨率检查 (1280x960最低)
  - 帧率检查 (25fps最低)
  - FOV检查 (水平≥70°)
  - 动态范围 (≥100dB)
  
- [ ] 场景自适应分析
  - 前视ADAS: 重点评估车道线清晰度
  - 环视: 畸变校正质量评估
  - DMS: 红外响应/面部清晰度
  - 座舱: 色彩还原准确性

- [ ] 夜间成像专项
  - 暗光噪声评估 (< 10 lux)
  - HDR vs 非HDR对比
  - LED闪烁抑制评估

### 评分标准
```python
ADAS_FRONT_SCORE = {
    'excellent': (90, 100),   # 4K + 60fps + HDR
    'good': (75, 90),         # 1080P + 30fps + HDR
    'acceptable': (60, 75),    # 1080P + 30fps
    'poor': (0, 60)           # 分辨率或帧率不足
}
```

---

## Phase 4: 知识库扩展

### 目标
构建更全面的ISP调优知识库

### 实现内容
- [ ] ISP模块诊断规则扩展
  - [ ] Bayer/Demosaic问题
  - [ ] 降噪参数优化
  - [ ] 锐化伪影诊断
  - [ ] 色彩空间转换问题
  
- [ ] 场景化参数推荐
  - [ ] 晴天/阴天/夜景参数
  - [ ] 运动/静止场景
  - [ ] 低光/高光环境
  
- [ ] 问题根因分析
  - [ ] 症状 → 模块映射
  - [ ] 多症状联合诊断
  - [ ] 优先级排序

---

## Phase 5: 测试覆盖率提升

### 目标
关键模块测试覆盖率达到80%+

### 实现内容
- [ ] `test_ai_quality_scorer.py` - 质量评分测试
- [ ] `test_hdr_processor.py` - HDR处理测试
- [ ] `test_automotive_analyzer.py` - 车载分析测试
- [ ] `test_tuning_knowledge.py` - 知识库测试
- [ ] `test_brisque.py` - BRISQUE特征测试

### 测试策略
- 单元测试：每个函数独立测试
- 合成测试：模拟真实场景
- 边界测试：异常输入处理

---

## Phase 6: CLI/API增强

### 实现内容
- [ ] 交互式调参模式
  ```bash
  isp-agent tune --interactive --scene automotive
  ```
  
- [ ] 报告生成增强
  - [ ] HTML报告模板
  - [ ] PDF导出支持
  - [ ] 多图像对比报告
  
- [ ] API增强
  - [ ] WebSocket实时流分析
  - [ ] 批量任务队列
  - [ ] 结果缓存

---

## 技术债务清理

- [ ] 统一异常处理
- [ ] 日志规范化
- [ ] 类型注解完善
- [ ] 文档字符串补全

---

## v0.4 里程碑

| 日期 | 里程碑 |
|------|--------|
| Day 1-2 | BRISQUE特征提取核心实现 |
| Day 3-4 | HDR Tone Mapping算法实现 |
| Day 5-6 | 车载分析深度改进 |
| Day 7-8 | 知识库扩展 |
| Day 9-10 | 测试编写 |
| Day 11-12 | CLI/API增强 |
| Day 13-14 | 文档完善 + 集成测试 |

---

## 资源需求

- Python 3.9+
- OpenCV 4.x
- NumPy/SciPy
- scikit-learn (SVR for BRISQUE)
- rawpy (可选，RAW支持)
- pytest (测试框架)
