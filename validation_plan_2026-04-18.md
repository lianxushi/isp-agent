# ISP-Agent 实际使用验证方案
> 日期: 2026-04-18
> 负责人: 川流 + 图灵

---

## 一、验证目标

使用真实开源 ISP 项目验证 ISP-Agent 的实际效果

## 二、开源 ISP 项目筛选

| 项目 | Stars | 语言 | 特点 | 适用场景 |
|------|------|------|------|----------|
| **openISP** | 1395 | C | 通用 ISP pipeline | 通用图像处理验证 |
| **Infinite-ISP** | 319 | Python | 模块化 + TuningTool | **★ 重点验证** |
| **ISPLab** | 180 | C++ | 学术风格 | 算法对比参考 |
| **AISP** | 562 | Python | AI-based (NTIRE挑战) | AI vs 传统对比 |

## 三、验证方案

### 3.1 Infinite-ISP（主验证对象）

**选择原因**：Python 实现易测试 + 有 TuningTool + 模块化设计

**验证步骤**：
1. 克隆部署 Infinite-ISP
2. 使用标准 RAW 测试集（建议用项目自带样本）
3. 运行完整 ISP Pipeline（demosaic → denoise → AWB → CCM → gamma → sharpen）
4. 使用 ISP-Agent 分析输出图像质量
5. 对比调参前后的 BRISQUE/NIQE 分数

**测试指标**：
- BRISQUE 分数
- NIQE 分数
- 处理速度
- 输出图像质量主观评价

### 3.2 openISP（辅助验证）

**验证步骤**：
1. 编译 openISP
2. 使用标准测试图像
3. 对比 ISP-Agent 分析结果

## 四、分工安排

| 任务 | 负责人 | 状态 |
|------|--------|------|
| 项目调研 + 环境准备 | 川流 | 🔄 进行中 |
| Infinite-ISP 部署 + 测试 | 图灵 | ⏳ 待开始 |
| openISP 编译 + 测试 | 川流 | ⏳ 待开始 |
| 交叉验证结果 | 川流 + 图灵 | ⏳ 待开始 |
| 生成测试报告 | 川流 | ⏳ 待开始 |

## 五、测试数据

优先使用项目自带测试样本：
- Infinite-ISP: `data/` 目录
- openISP: 项目测试图像

## 六、预期交付物

1. 各项目部署文档
2. 测试结果截图/数据
3. BRISQUE/NIQE 分数对比表
4. 问题记录和改进建议

---

## 七、时间节点

- 10:00 - 完成项目筛选和环境准备
- 12:00 - Infinite-ISP 测试完成
- 15:00 - openISP 测试完成
- 17:00 - 交叉验证完成
- **19:00 - 向石先生汇报结果**
