# ISP-Agent 项目

基于LLM的ISP图像/视频处理助手，为石先生(ISP算法工程师)提供智能图像质量分析和调参建议。

## 🤖 Agent协作三条铁律（所有Agent必须遵守）

1. **遇到错误必须先报告再继续** — 不许默默失败或绕过
2. **禁止用相似任务替代失败任务** — 不许凑合
3. **用序号列出的指令必须逐条确认完成** — 不许漏做跳做

## 技术栈

- **语言**: Python 3.9+ / C++ (部分模块)
- **图像处理**: OpenCV, rawpy, numpy, scipy, Pillow
- **LLM**: MiniMax API (主力), OpenAI GPT, Anthropic Claude (备用)
- **工具**: Docker, REST API, CLI
- **测试**: pytest
- **文档**: Markdown (docs_v2/ 为最新版本)

## 项目结构

```
isp-agent/
├── src/
│   ├── agent/          # Agent核心 (LLM调用、对话管理)
│   ├── api.py          # REST API服务
│   ├── cli_enhanced.py # 增强CLI
│   ├── image_analyzer.py
│   ├── isp_compare/    # ★ ISP版本对比框架 (v1.0, 2026-03-28)
│   │   ├── core/       # 核心对比引擎
│   │   ├── models/     # 数据模型
│   │   ├── reports/    # 报告生成
│   │   ├── tools/      # 工具集
│   │   └── utils/      # 工具函数
│   ├── processors/     # 图像/视频处理器
│   ├── tools/          # 工具模块
│   └── utils/          # 通用工具
├── tests/              # 测试用例
├── docs/               # 项目文档 (旧版)
├── docs_v2/            # ★ 项目文档 (最新)
│   ├── 01-需求分析/    # 需求规格
│   ├── 02-功能模块设计/ # ISP版本对比等模块设计
│   ├── 03-详细设计/    # 详细设计
│   └── 05-技能设计/    # Agent技能定义
├── config/             # 配置文件
├── scripts/            # 辅助脚本
├── venv/               # Python虚拟环境
├── requirements.txt
├── docker-compose.yml
└── Dockerfile
```

## 核心能力

### 1. ISP版本对比框架 (★ 当前开发重点)

**模块定位**: 自动化批量对比ISP版本差异，减少Imatest人工操作

**核心指标**:
- BRISQUE (盲图像质量评分)
- HDR动态范围
- 色彩准确性
- 噪声水平
- 边缘增强 (EE)
- 局部色调映射 (LTM)
- ISO16505合规

**数据格式**:
- Comp12 RAW (12-bit, RGGB/BGGR/GRBG/GBRG)
- DNG/RAW格式支持

**输出**: HTML对比报告 + 调参建议

### 2. 图像质量分析
- 直方图分析、噪声估计、动态范围计算
- ISP参数诊断 (Gamma, EE, LTM)
- RAW域处理

### 3. 视频处理
- 抽帧、质量检测、基本信息分析

### 4. 智能问答
- 中文自然语言ISP领域问答
- 调参建议生成

## 工作流程

### 开发新功能
1. 查看 docs_v2/ 了解最新设计
2. 在 src/ 对应模块开发
3. 添加 tests/ 测试用例
4. 更新 docs_v2/ 文档

### 运行测试
```bash
cd ~/.openclaw/workspace/isp-agent
source venv/bin/activate
pytest tests/ -v
```

### 运行CLI
```bash
python -m src.main
# 或
python src/cli_enhanced.py
```

### 启动API服务
```bash
docker-compose up -d
# 或
python -m src.api
```

## 项目管理

- **版本**: v0.4 (开发中)
- **最新commit**: feat: ISP Compare framework v1.0 - Core modules (2026-03-28)
- **文档分支**: docs_v2/ (最新设计文档)
- **旧文档**: docs/ (参考，不更新)

## 📝 代码提交规范（强制执行）

### ⚠️ 每次完成任何任务后必须立即提交！

**规则**：
1. 完成任何功能开发 → **立即 git add + git commit**
2. 提交信息格式：`type: 简短描述`
   - `feat: 新功能`
   - `fix: 修复bug`
   - `docs: 文档更新`
   - `test: 测试用例`
3. **不允许积累变更不提交**（最多当天结束前必须提交）
4. 每次提交前检查 `git status`

### 自动检查
在 isp-agent 目录执行任何操作后，必须运行：
```bash
git add -A
git status --short
git commit -m "type: 描述"
```

### 提交示例
```bash
git add -A
git commit -m "feat: 添加BRISQUE特征提取模块"
```

---

## 负责人

- **石连旭 (Boss)**: ISP算法工程师，项目发起人
- **川流 (AI助手)**: 文档开发、代码辅助
- **图灵 (Dev Subagent)**: 负责代码开发，**每次完成后必须立即提交**

## 注意事项

- RAW图像处理: 使用 rawpy 库，输入16bit PNG优先
- BRISQUE计算: scipy依赖，使用 math.gamma 替代 np.math.gamma
- Docker部署: 使用 docker-compose.yml
- API认证: MiniMax API Key 在环境变量或 config/
