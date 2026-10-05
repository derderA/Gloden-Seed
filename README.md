# 智诊AI平台应用

智诊AI是一个面向 K12 作业诊断场景的教学分析平台，围绕“上传作业图片 -> OCR 结构化 -> 知识图谱回溯 -> 大模型归因 -> 3D 干预展示 -> 历史聚合”的闭环构建。

当前项目已经不是纯演示壳子，而是具备以下真实能力：

- 支持 `demo` / `tencent` 两类 OCR Provider
- 支持 `mock` / `ollama` / `deepseek` 三类诊断 Provider
- 支持 `demo` / `tripo` 两类 3D 干预 Provider
- 默认使用 `K12-KGraph` 合并图谱进行前置依赖回溯
- 支持缺失知识点的补图建议与写回
- 支持跨题历史聚合，用于区分偶然失误与稳定盲区

## 1. 项目结构

```text
homework/
├── app/
│   ├── api/                  # FastAPI 路由
│   ├── data/                 # 知识图谱、历史记录
│   ├── generated/            # 生成结果目录
│   ├── services/             # OCR/诊断/图谱/3D 服务
│   ├── static/               # 前端 CSS/JS
│   ├── templates/            # 页面模板
│   ├── uploads/              # 上传缓存与 OCR 裁图
│   ├── config.py             # 配置读取
│   ├── main.py               # 应用入口
│   └── models.py             # 数据模型
├── tests/
├── .envs                     # 本地配置文件
├── README.md
├── 项目报告.md
└── requirements.txt
```

## 2. 核心能力

### 2.1 OCR 结构化

- `demo`：本地演示数据，适合调试页面和链路
- `tencent`：真实调用腾讯云教育 OCR `SubmitQuestionMarkAgentJob`
  - 已接入 `TC3-HMAC-SHA256` 签名
  - 已支持 `KnowledgePoints / TrueAnswer / StepCorrection / DisableAnswerAnalysis / OutputSubQuestionsAndCoords / UseCoordAssist`
  - PDF 自动携带 `PdfPageNumber=1`
  - 当腾讯只返回题块坐标时，会进一步尝试裁图并结合 LLM 做补全

### 2.2 知识图谱回溯

- 默认图谱：`app/data/knowledge_graph_k12_merged.json`
- 数据基础：
  - `K12-KGraph` 官方样例图谱
  - 项目补充节点与边
- 运行机制：
  - 先根据题目知识点做归一化
  - 再从图谱中回溯前置依赖路径
  - 生成用于前端展示的 DAG 子图
- 安全约束：
  - LLM 只做回溯辅助，不允许替代图谱主路径
  - 如果当前知识点未命中图谱，不再采纳 LLM 虚构路径

### 2.3 缺失知识点补图

当当前题目的知识点无法命中已有图谱时：

1. 前端会在“知识图谱回溯”页提示当前暂无 DAG
2. 用户可选择是否把该知识点加入现有图谱
3. 后端会调用 LLM 生成候选：
   - 节点名称
   - 描述
   - 别名
   - 前置依赖
   - 相关连接
   - 从属关系
4. 用户确认后写回当前图谱文件
5. 页面立即刷新新的回溯 DAG

### 2.4 根因诊断

- `mock`：启发式诊断
- `ollama`：本地模型诊断
- `deepseek`：云端 API 诊断

诊断目标包括：

- 区分“理解层 / 执行层”
- 定位前置漏洞知识点
- 输出带置信度的错因分布
- 给出可执行的补救建议
- 结合历史记录做跨题聚合

如果大模型调用失败，系统会自动回退到启发式诊断，并把失败原因显示给前端。

### 2.5 3D 干预展示

- `demo`：本地 Three.js 交互场景
- `tripo`：真实调用 Tripo 文生 3D 接口

当前已实现：

- Tripo `text-to-model` 任务提交
- Tripo 任务轮询
- 模型下载链接与预览图回传
- Tripo 失败时自动回退到本地 Three.js 场景

本地 Three.js 已内置三类教学干预场景：

- `fraction_blocks`：分数通分
- `cylinder_slices`：圆柱体积
- `magnet_field`：磁感线方向

## 3. 运行环境

建议使用你当前项目约定的 `myenv-3.9` 环境：

```bash
conda activate myenv-3.9
cd /home/derder/homework
python -m pip install -r requirements.txt
```

也可以直接使用解释器绝对路径：

```bash
/home/derder/miniconda3/envs/myenv-3.9/bin/python -m pip install -r /home/derder/homework/requirements.txt
```

## 4. 配置说明

项目只从根目录的 `.env` 和 `.envs` 读取配置，且以 `.envs` 为最终覆盖源。前端不输入任何 API Key。

示例配置：

```env
APP_NAME=智诊AI
APP_HOST=0.0.0.0
APP_PORT=8000

DEFAULT_OCR_PROVIDER=demo
DEFAULT_LLM_PROVIDER=mock
DEFAULT_MODEL_PROVIDER=demo

OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b

DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your_deepseek_key

TENCENT_SECRET_ID=your_tencent_secret_id
TENCENT_SECRET_KEY=your_tencent_secret_key
TENCENT_OCR_REGION=ap-beijing
TENCENT_OCR_VERSION=2018-11-19

TRIPO_API_KEY=your_tripo_api_key
TRIPO_BASE_URL=https://openapi.tripo3d.com/v3
TRIPO_MODEL_VERSION=v3.1-20260211
TRIPO_POLL_INTERVAL_SECONDS=2
TRIPO_TIMEOUT_SECONDS=120

KNOWLEDGE_GRAPH_PATH=app/data/knowledge_graph_k12_merged.json
```

## 5. 启动方式

```bash
cd /home/derder/homework
/home/derder/miniconda3/envs/myenv-3.9/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

启动后访问：

```text
http://127.0.0.1:8000
```

## 6. 使用流程

1. 上传作业图片
2. 选择或确认 Provider
3. 输入题目类型提示
4. 发起诊断
5. 查看：
   - OCR 结构化结果
   - 根因诊断
   - 知识图谱回溯 DAG
   - 历史聚合盲区
   - 3D 干预结果

如果“知识图谱回溯”页没有 DAG，系统会提示是否将该知识点加入当前图谱。

## 7. 图谱资源说明

### 7.1 当前默认图谱

默认知识库已经切换到：

`app/data/knowledge_graph_k12_merged.json`

这份图谱包含：

- `K12-KGraph` 官方样例节点与依赖边
- 项目补充的分数、圆柱体积、磁场方向等节点
- 运行中新增写回的节点与关系

### 7.2 支持的图谱格式

支持两类格式：

1. 项目原生格式

```json
{
  "nodes": [
    {
      "name": "分数通分",
      "subject": "数学",
      "prerequisites": ["分数意义", "最小公倍数"]
    }
  ]
}
```

2. K12-KGraph 风格格式

```json
{
  "nodes": [
    {"id": "c1", "type": "Concept", "name": "分数意义", "subject": "数学"},
    {"id": "s1", "type": "Skill", "name": "分数通分", "subject": "数学"}
  ],
  "edges": [
    {"source": "c1", "target": "s1", "relation": "prerequisites_for"}
  ]
}
```

### 7.3 外部图谱替换

如果你有更完整的 K12-KGraph 导出文件，可以直接在 `.envs` 中替换：

```env
KNOWLEDGE_GRAPH_PATH=/你的图谱文件路径.json
```

重启应用后自动生效。

## 8. API 与页面模块

主要接口：

- `GET /api/health`
- `GET /api/history/summary`
- `POST /api/pipeline/run`
- `POST /api/pipeline/stream`
- `POST /api/graph/suggest`
- `POST /api/graph/add`

前端主要模块：

- 诊断任务
- 错题结构化
- 根因诊断
- 知识图谱回溯
- 跨题聚合盲区
- 诊断过程
- 3D 干预展示

## 9. 测试

运行全部测试：

```bash
cd /home/derder/homework
/home/derder/miniconda3/envs/myenv-3.9/bin/python -m pytest
```

如果你只想快速验证图谱与诊断主链路，可以跑：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/derder/miniconda3/envs/myenv-3.9/bin/python -m pytest -q tests/test_graph_service.py tests/test_pipeline.py
```

## 10. 当前实现边界

当前系统已经可以稳定演示与迭代，但仍有这些边界：

- `demo` OCR 仅覆盖少量内置题型
- 腾讯 OCR 在某些图片上仍可能只返回坐标，需要依赖 LLM 二次补全
- 新增知识点时，目前是“LLM 候选 + 用户确认”模式，还没有做人工可视化编辑器
- Tripo 返回的模型链接通常是时效性的，需要尽快查看或下载

## 11. 后续建议

- 为图谱补充更细粒度的学段、章节、教材版本过滤
- 给“补图建议”增加人工编辑能力
- 将历史聚合进一步细化到错误步骤模式
- 为 Tripo 结果增加本地缓存与 GLB 在线预览
- 增加教师看板、班级统计、学生画像等上层功能

## 12. 参考资源

- K12-KGraph GitHub: https://github.com/haolpku/K12-KGraph
- K12-KGraph Project Page: https://haolpku.github.io/K12-KGraph-page/
- K12-KGraph Dataset: https://huggingface.co/datasets/lhpku20010120/K12-KGraph
