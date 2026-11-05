# 通用生图插件 (Image Generation)

![:name](https://count.getloli.com/@astrbot_plugin_image_generation?name=astrbot_plugin_image_generation&theme=miku&padding=7&offset=0&align=top&scale=1&pixelated=1&darkmode=auto)

面向 AstrBot 的通用图像生成插件，支持多供应商、文生图、图生图、LLM 工具调用、预设/人设、任务队列、限流、历史记录和安全审核。

## 功能

- **多供应商与多适配器**：内置 Gemini、OpenAI Chat/Images、火山方舟、Gitee AI、硅基流动、Agnes AI、Grok、Codex Responses、ModelScope、自定义 HTTP 和 ComfyUI 工作流；每个供应商可独立配置接口、代理、密钥、模型和能力，多个 Key 自动轮询。
- **文生图与图生图**：自动收集当前消息、引用消息、@ 用户头像、预设参考图和人设参考图；模型不支持图生图时自动忽略参考图。
- **引用/转发取词**：引用消息或合并转发卡片会取第一层文字并入提示词（最多 1000 字符），不取其中的图片。
- **任务体系**：支持多图生成、请求级并发、任务级队列和分批发送；可查看/取消任务，任务历史与生成图片均可持久化。
- **LLM 工具**：`generate_image`、`query_image_presets`、`manage_image_tasks`，可选启用 `edit_image_presets`。
- **预设与人设**：预设模板支持提示词、宽高比、分辨率和参考图；支持多人设拼接、人设参考图，可选正文名称匹配；旧版 `名称:提示词` 预设仍可正常读取。
- **限制与审核**：会话黑名单、管理员/白名单绕过、会话频率限制、群聊与个人双层每日额度、参考图大小限制；支持屏蔽词与 AI 提示词/图片审核。
- **插件间公共 API**：其他插件可提交任务、查询状态、取消任务、等待结果并获取本地图片路径。

## 群聊与个人每日额度

在“使用限制”中可独立启用“群聊每日额度”和“个人每日额度”。例如群聊 `30` 张、个人 `3` 张：同群每人各有 3 张，同时全群合计不能超过 30 张；同一用户在不同群独立统计。启用两层时必须同时满足，按请求数量预留，按实际保存图片数量结算。

旧版每日额度和已有 UMO 用量自动继承为群聊额度，新增个人额度默认关闭。为避免升级后私聊无限制，开启群聊额度也保留私聊的旧会话限额；私聊可同时受个人额度限制。两项均关闭才是不设每日额度。管理员和 UMO 白名单仍绕过限制并记录用量。

公共 API 的用户请求应传 `sender_id=event.get_sender_id()`；仅传 UMO 的旧调用继续按会话限额处理，不会虚构个人身份。详见[配置说明](docs/configuration.md#使用限制配置)和[公共 API 文档](docs/public-api.md#调用身份与额度)。

## 命令

| 命令 | 说明 |
| :--- | :--- |
| `/生图 <预设/人设...> [额外提示词] [数量]` | 生成图片；末尾单独数字为数量，未写提示词时可使用引用/转发文字。 |
| `/生图任务` | 查看当前会话进行中的任务（带编号）。 |
| `/生图任务 <编号或任务ID>` | 查看任务详情。 |
| `/生图取消 <编号或任务ID>` | 取消排队或运行中的任务。 |
| `/生图模型` | 查看可用模型和当前模型。 |
| `/生图模型 <序号>` | 切换模型并立即刷新运行时配置。 |
| `/预设` | 查看所有预设和人设。 |
| `/预设 添加 <名称:内容>` | 添加或覆盖预设，支持中英文冒号。 |
| `/预设 删除 <名称>` | 删除预设。 |

## 适配器

| 适配器 | 接口 | 文生图 | 图生图 | 尺寸 | 说明 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `gemini` | Gemini 原生 `generateContent` | ✅ | ✅ | ✅ | Gemini 原生图像生成接口。 |
| `openai_chat` | OpenAI 兼容 `chat/completions` | ✅ | ✅ | ❌ | 通用 Chat 生图，可配置提示词前缀、输出模态和额外请求体。 |
| `openai` | Images `/v1/images/generations`、`/edits` | ✅ | ✅ | ✅ | GPT Image 2 / 2.5；支持参考图编辑与 1K/1.5K/2K/4K 分辨率。 |
| `volcengine_ark` | 方舟 `/api/v3`、Agent Plan `/api/plan/v3` | ✅ | ✅ | ✅ | Seedream 5.0 Pro / Flash，支持两种接口模式。 |
| `gitee_ai` | `/v1/images/generations`、`/edits` | ✅ | ✅ | ✅ | 可自动或手动选择生成/编辑接口。 |
| `siliconflow_adapter` | SiliconFlow Images API | ✅ | ✅ | ✅ | 支持 Kolors、Qwen-Image 等，可配置反向提示词、步数。 |
| `agnes_ai` | Agnes AI Images API | ✅ | ✅ | ✅ | `agnes-image-2.0/2.1/2.5-flash`，支持国际版/中国版。 |
| `grok` | xAI Images API | ✅ | ✅ | ✅ | 单图用 `image`，多图用 `images`（最多 3 张）。 |
| `codex_responses` | Codex / OpenAI Responses API | ✅ | ✅ | ❌ | 固定 Responses 请求，仅支持同步结果。 |
| `modelscope` | API-Inference + 任务轮询 | ✅ | 按配置 | 按配置 | 异步提交与轮询，详见 [ModelScope 配置](docs/modelscope.md)。 |
| `custom_http` | 自定义 HTTP JSON 接口 | ✅ | ✅ | ✅ | 高级模板，详见 [自定义 HTTP 配置](docs/custom-http.md)。 |
| `comfyui` | ComfyUI `/prompt` + `/history` | ✅ | 按配置 | 按配置 | 上传 API 格式工作流并轮询结果，详见 [ComfyUI 配置](docs/comfyui.md)。 |

> 能力以“模型能力”配置为准。未勾选的能力不会在请求中使用，LLM 工具参数也会按当前能力动态隐藏。

## 文档
> 具体配置细节见如下文件（如在AstrBot打不开，可以到仓库查看）
- [配置说明](docs/configuration.md)：所有配置项、默认值、占位符与适配器专属参数。
- [使用说明](docs/usage.md)：尺寸参数识别、引用/转发取词、任务体系。
- 接口配置：[ModelScope](docs/modelscope.md) · [Codex Responses](docs/codex-responses.md) · [自定义 HTTP](docs/custom-http.md) · [ComfyUI](docs/comfyui.md)

## 使用示例

```text
/生图 一只在森林里野餐的兔子
/生图 手办化 看板娘 夜景 微笑 3
/生图 猫 16:9 2K 3
```

尺寸参数识别、引用/转发取词和任务管理规则见 [使用说明](docs/usage.md)。

## 插件间公共 API

`plugin.public_api` 提供 `submit_generation_task`、`get_generation_task`、`cancel_generation_task`、`wait_generation_result` 和 `generate_image_files`。它复用本插件的任务、审核、参考图与保存流程，但不会主动发送图片。详见 [公共 API 文档](docs/public-api.md)。

## 开发

PR 请提交到 `dev` 分支，不要直接提交到主分支。

### 页面构建

WebUI 页面使用 Vue 3 + Vuetify 3，源码位于 `frontend/`，构建产物位于 `pages/dashboard/`。

修改 `frontend/` 后必须重新构建并提交产物，插件安装流程不会执行构建：

```bash
cd frontend
npm install
npm run build
```

页面运行在 sandbox iframe 中，所有请求都必须经 `window.AstrBotPluginView`（见 `frontend/src/bridge.js`）；主题色从 `frontend/src/theme/index.js` 复制自 AstrBot 主站，主站主题变更后需要同步。图标使用 `@mdi/js`，模板中必须写成 `$alias` 形式（如 `icon="$refresh"`），可用别名见 `frontend/src/theme/index.js`。

### 离线额度测试

安装 `requirements.txt` 与 `pytest` 后执行 `python -m pytest tests -q`。测试 mock AstrBot 和模型结果，直接运行插件的真实额度、任务、命令、LLM 工具与公共 API 代码，不连接生图服务。基础静态检查可执行 `ruff check --isolated --select E4,E7,E9,F,I main.py core tests`。
