# voiceprint-focus

实时流式**目标说话人提取**（Target Speaker Extraction, TSE）：给一段注册语音作为声纹提示，
从单通道 16 kHz 麦克风信号里持续分离出目标说话人，其余人声与环境噪声被抑制。端到端因果、
32 ms 分块推理，面向 CPU 实时场景。

本仓库是应用的代码侧，包含两个可独立使用的形态：

| 形态 | 入口 | 说明 |
|---|---|---|
| Web 应用 | `backend/server.py` + `frontend/` | FastAPI + WebSocket + ONNX Runtime（CPU），React 前端；支持录制注册、文件注册、实时开流与离线文件处理 |
| 虚拟麦克风 | `python -m vmic`（Windows） | 采集真实麦克风 → 流式聚焦 → 写入虚拟声卡，供 Discord / Zoom / OBS / 游戏语音选择为麦克风 |

模型为因果 BSRNN + 帧级 TF-Map 条件化，导出为两个 ONNX：`enroll.onnx`（声纹编码）与
`step.onnx`（32 ms 流式分离）。

实测效果披露（评测口径、样本量、置信区间、失败模式与局限）见
**[RESULTS.md](RESULTS.md)**；训练与评测方法、推理链路、交付形态的全景图见
**[docs/methodology.svg](docs/methodology.svg)**。

## 仓库结构

```text
├── backend/            # FastAPI 服务、流式引擎、STFT/iSTFT、模型定义、ONNX 导出与校验
│   ├── server.py       # HTTP / WebSocket 服务入口
│   ├── engine.py       # 流式会话、注册质检、旁路与后处理
│   ├── dsp.py          # STFT / iSTFT / overlap-add
│   ├── model_def.py    # 模型定义（WeSep BSRNN 的因果化推理裁剪版，见文件头）
│   ├── export_onnx.py  # 从训练 checkpoint 导出 ONNX（需要 torch）
│   ├── campp.py        # CAM++ 注册旁路观察（advisory only，可禁用）
│   └── tools/          # 演示素材生成、引擎离线校验
├── frontend/           # React + Vite 前端
├── vmic/               # Windows 虚拟麦克风 CLI
├── tests/              # 单测（不依赖权重与演示素材）
└── models/             # 权重目录（不入库，见下）
```

## 环境要求

- Python 3.10+
- Node.js 18+（仅构建/开发前端时需要）
- Windows（仅 `vmic` 形态需要；Web 形态跨平台）

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

可选依赖：

- 导出 ONNX 或跑离线校验：`pip install torch`
- 启用 CAM++ 注册旁路观察：`pip install sherpa-onnx`（缺失时自动禁用，不影响主链路）

## 获取模型权重

**权重不在本仓库中。** 它与代码分开、按 **Research Use Only** 单独发布，见 GitHub Release：

- [x16-weights-v1](https://github.com/Ximo0519/voiceprint-focus/releases/tag/x16-weights-v1)
  包含 `step.onnx`、`enroll.onnx`、`export_report.json`、`LICENSE-WEIGHTS`（权重许可全文）、
  `NOTICE` 与 `SHA256SUMS.txt`。

拿到权重后，放到以下位置：

```text
models/
├── x16_stream/
│   ├── step.onnx          # 流式分离模型（约 65.5 MiB）
│   ├── enroll.onnx        # 声纹编码模型（约 1.45 MiB）
│   └── export_report.json # 导出校验报告（服务启动时读取）
└── campplus/              # 可选：CAM++ 旁路观察
    └── 3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
```

CAM++ 权重为第三方模型，可从 ModelScope 的
[`iic/speech_campplus_sv_zh-cn_16k-common`](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common)
获取，并遵守其自身许可。

## 运行

### Web 应用

```powershell
# 1) 构建前端（生成 frontend/dist，由后端静态托管）
cd frontend
npm install
npm run build
cd ..

# 2) 启动服务
python backend/server.py
# 打开 http://127.0.0.1:8000
```

前端开发模式（Vite 代理到 `127.0.0.1:8000`）：

```powershell
cd frontend
npm run dev
```

主要接口：

- `POST /api/enroll/file`：上传音频注册声纹
- `GET  /api/enroll/list`：已注册声纹列表
- `POST /api/process/file`：离线文件处理（流式返回）
- `WS   /ws`：实时开流（协议见 `backend/server.py` 文件头）
- `GET  /api/health`：模型、线程、绑核与运行参数

### 虚拟麦克风（Windows）

```powershell
python -m vmic devices                 # 列出音频端点
python -m vmic enroll                  # 录制 3–6 秒注册声纹
python -m vmic run --output "CABLE Input"
```

输出端点选虚拟声卡（如 VB-CABLE 的 `CABLE Input`），再到 Discord / Zoom / OBS 里把麦克风
选成它的 `CABLE Output`。参数与默认值见 `python -m vmic run --help`。

## 没有权重时哪些功能跑不起来

- `python backend/server.py` **会启动失败**：启动阶段必须加载
  `models/x16_stream/enroll.onnx`、`step.onnx` 与 `export_report.json`，缺失会直接抛
  `FileNotFoundError`。
- 前端可以单独构建并在浏览器打开，但所有注册、开流、文件处理都依赖后端与权重，实际不可用。
- `python -m vmic run` 会提示“缺少模型文件”并退出；`devices` / `profiles` 仍可查看端点与档案。
- CAM++ 旁路观察是**可选**的：其模型或 `sherpa-onnx` 缺失时只会在 `/api/health` 里标记为
  禁用，不影响注册与分离主链路。
- 演示素材（`backend/assets/demo/`）不随仓库分发，`/api/demo/list` 会返回空列表；可用
  `backend/tools/make_demo_assets.py` 基于自己的音频生成。

## 数据与隐私

- 注册音频默认落盘到 `backend/enrollments/`（WAV + JSON 元数据），声纹嵌入存于其
  `registrations/` 子目录。两者均已在 `.gitignore` 中排除，**不要提交**。
- 关闭留档：启动前设置 `VF_ENROLL_SAVE=0`；改目录：`VF_ENROLL_DIR=<path>`。
- 采集录音与 CAM++ 诊断分别写入 `backend/captures/`、`backend/campp_diag/`，均不入库。
- 声纹属敏感个人信息。部署或演示时必须取得说话人单独同意，并遵守所在司法辖区的
  个人信息保护与生物识别相关规定。

常用环境变量：

| 变量 | 默认 | 作用 |
|---|---|---|
| `VF_ENROLL_SAVE` | `1` | 是否落盘注册音频 |
| `VF_ENROLL_DIR` | `backend/enrollments` | 注册音频与声纹档案目录 |
| `VF_CPU_AFFINITY` | `off` | CPU 绑核策略，见 [`backend/CPU_AFFINITY.md`](backend/CPU_AFFINITY.md) |
| `VF_CAMPP` | `1` | 设为 `0` 关闭 CAM++ 旁路观察 |
| `VF_CAMPP_DIAG_DIR` | `backend/campp_diag` | CAM++ 诊断输出目录 |
| `VF_POST_DENOISE` | `1` | 后置降噪默认开关 |

## 许可 / License

本仓库的**代码**采用 [Apache License 2.0](LICENSE)；**模型权重**按 **Research Use Only**
单独发布，见 [GitHub Release x16-weights-v1](https://github.com/Ximo0519/voiceprint-focus/releases/tag/x16-weights-v1)，
其许可全文随权重包单独提供，禁止商业使用。

- 上游署名、第三方组件与数据集的条款见 [`NOTICE`](NOTICE)。
- `backend/model_def.py` 是 [wenet-e2e/WeSep](https://github.com/wenet-e2e/WeSep)
  （Apache-2.0，`wesep_2024` 布局）的衍生作品，已在文件头保留上游署名、SPDX 标识并逐项
  声明实质性修改（含 `band_comm` 双向改单向、因果 LSTM、`PerFrameNorm`、移除 utterance
  条件通路、新增 TF-Map-lite 分支等）。
- 本仓库**不包含**任何模型权重、训练数据、用户录音或第三方音频。

使用本项目即表示你已阅读并同意上述许可与使用限制；代码与权重均按“现状”提供，不附带担保。
