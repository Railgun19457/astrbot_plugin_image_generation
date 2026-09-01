# Codex Responses 接口配置

`codex_responses` 适配器用于接入采用 OpenAI Responses 风格的同步图片生成接口。它是专用模板：插件固定构造请求、发送 Bearer API Key，并识别 `image_generation_call` 中的图片结果，不需要手动配置请求体或图片路径。

当前版本支持：

- `POST` 同步请求，接口路径默认为 `/codex/responses`，可改为 `/v1/responses` 等
- `Authorization: Bearer <API Key>` 认证
- 同步 JSON 响应
- 文生图和参考图驱动的图像编辑
- 标准 `output[*].type == "image_generation_call"` 的 `result` 图片字段
- 纯 Base64、`data:image/...;base64,...` 和 HTTP(S) 图片 URL
- 兼容 `result_b64`、`b64_json`、`base64` 以及 `data[*]` 包装的图片结果
- API Key 轮换、代理、超时、重试与现有任务发送流程

暂不支持：

- 宽高比与分辨率参数
- 流式 SSE 响应
- 返回任务 ID 后需要轮询的异步接口
- 下载图片 URL 时额外转发 API Key 或自定义下载请求头

## 快速配置

1. 在“图像模型供应商”中新增 **Codex Responses 接口**。
2. 填写供应商名称，例如 `宝宝AI`。
3. 在“接口地址”填写服务根地址，例如 `https://baobao-ai.com`。
4. 在“接口路径”填写实际路径；默认 `/codex/responses`。标准 OpenAI Responses 填 `/v1/responses`。
5. 在“API 密钥”填写宝宝 AI API Key。
6. 在“可用模型列表”填写服务实际支持的模型，例如 `gpt-5.6-terra`。
7. 在“生图模型”中选择 `宝宝AI/gpt-5.6-terra`。

实际请求地址由接口地址和接口路径直接拼接，不再猜测或改写路径：

| 接口地址 | 接口路径 | 实际请求地址 |
| :--- | :--- | :--- |
| `https://baobao-ai.com` | `/codex/responses` | `https://baobao-ai.com/codex/responses` |
| `https://api.example.com` | `/v1/responses` | `https://api.example.com/v1/responses` |
| `https://chatgpt.com/backend-api` | `/codex/responses` | `https://chatgpt.com/backend-api/codex/responses` |

接口地址必须是 `http://` 或 `https://` URL，且不能包含查询参数或片段。不要把完整接口路径填进接口地址。

## 固定请求格式

每张图片都会发起一个独立请求；插件现有的多图任务会按请求级并发配置调度这些请求。

请求头：

```http
Authorization: Bearer <API Key>
Content-Type: application/json
```

请求体固定为：

```json
{
  "model": "gpt-5.6-terra",
  "input": "生成一张图片：白色背景中央有一个小的纯蓝色实心圆，不要文字。",
  "tools": [
    {
      "type": "image_generation",
      "output_format": "png"
    }
  ]
}
```

其中 `input` 使用插件经过预设、人设和附加提示词合并后的最终文本。文本生图不会额外添加 `messages`、`prompt`、`n`、`stream`、`modalities`、`size`、参考图或尺寸字段。工具声明会固定带 `output_format: "png"`，以匹配已验证的宝宝 AI 请求。

## 图像编辑请求

当用户发送、引用或通过 LLM 工具/公共 API 提供参考图时，插件会复用既有的参考图收集、大小校验、文件类型校验、去重和转换链路，并把每张 `ImageData` 按原顺序编码为 `data:<mime>;base64,...`。此时请求使用 Responses 多模态 `input`：

```json
{
  "model": "gpt-5.6-terra",
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "将这张图片改成赛博朋克夜景风格，保留人物姿势和画面构图。"
        },
        {
          "type": "input_image",
          "image_url": "data:image/jpeg;base64,<YOUR_BASE64_IMAGE>"
        }
      ]
    }
  ],
  "tools": [
    {
      "type": "image_generation",
      "output_format": "png"
    }
  ]
}
```

这与宝宝 AI 的 `gpt-5.6-terra` 编辑请求和同步成功响应已经核验一致。输出仍从：

```text
response.output[i].type == image_generation_call
response.output[i].result
```

读取，`result` 是 Base64 编码的图片二进制。

- 直接发送或引用图片后，可使用 `/生图 <编辑要求>`。
- LLM 工具的 `reference_images`、`avatar_references` 和带图片的人设会进入同一条图生图链路。
- 插件公共 API 的 `reference_image_sources` 和 `reference_image_data` 也会进入同一条图生图链路。
- 多张参考图会全部按顺序作为 `input_image` 发送；宝宝 AI 已验证单张输入。若当前账号或模型限制多图，请只提供一张参考图，插件不会静默截断。

宽高比和分辨率当前仍不会加入 Codex 请求。

## 响应格式

首选解析的标准 Responses 响应形态：

```json
{
  "id": "resp_example",
  "status": "completed",
  "error": null,
  "output": [
    {
      "type": "reasoning"
    },
    {
      "type": "image_generation_call",
      "status": "generating",
      "revised_prompt": "...",
      "result": "<Base64 编码的 PNG 图像数据>"
    },
    {
      "type": "message",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "..."
        }
      ]
    }
  ]
}
```

插件会扫描所有 `output` 项，筛选 `type` 为 `image_generation_call` 的项目，再优先读取其 `result`。即使该调用项的 `status` 仍是 `generating`，只要同一响应已经给出可解码的 `result`，插件仍会将其作为图片结果处理。

`result` 可以是：

- 纯 Base64 图像数据（宝宝 AI 实测格式）
- `data:image/png;base64,...` 形式的 data URL
- 可公开下载的 `http://` 或 `https://` 图片 URL

为兼容常见中转实现，插件也会尝试 `result_b64`、`b64_json`、`base64`、`url`、顶层等价字段以及 `data[*]` 下的图片字段。

如果接口只返回任务 ID、`202 Accepted`、SSE 事件流，或者图片 URL 下载需要另一种认证方式，则此专用适配器不会完成图片生成；这类协议需要独立的异步/流式适配器。

## 能力边界

当前 provider 支持“文生图”和“图生图”。当有参考图时，插件会使用上文的多模态 `input_image` data URL；当没有参考图时，插件保留已验证的字符串 `input` 请求格式。

以下参数仍由统一任务执行器忽略，不会进入 Codex 请求：

- 宽高比
- 分辨率

请勿在服务端猜测或复用其他 OpenAI 接口的 `size` 或 `quality` 字段；待 Codex Responses 提供明确协议后再扩展。

## 排障

### 出现 `Server disconnected`，但任务最终成功

这不是本插件把请求超时缩短到约 150 秒。Codex provider 的“超时时间覆盖”会传给 `aiohttp.ClientTimeout(total=...)`；例如配置为 `300` 时，客户端总请求超时仍是 300 秒。`Server disconnected` 表示在收到 HTTP 响应前，服务端、显式代理、透明代理、负载均衡、CDN 或其他网络设备主动关闭了底层连接。

插件会将这种无 HTTP 状态码的连接错误视为可重试错误。因此日志可能先显示一次连接异常，随后由重试请求成功，任务最终仍正确保存并发送图片。新版日志会在“适配器请求”与“请求异常”中标出配置的超时秒数，便于确认实际生效值。

排查与解决顺序：

1. 在 Codex provider 中暂时清空“代理地址”后复测；如果问题消失，调整该代理的 HTTP read/idle timeout。
2. 在宝宝 AI、反向代理、CDN/WAF 或负载均衡侧，将上游/下游的 read、response 或 idle timeout 设为高于插件请求超时（300 秒应留出缓冲）。
3. 从运行 AstrBot 的同一台机器直接向服务发送同样请求，比较是否仍在约 150 秒断开，以区分插件外的网络链路。
4. 若服务无法维持同步长连接，应改用服务端提供的异步轮询或流式协议；当前专用适配器只支持同步最终响应。

不要盲目增加客户端 timeout 或无限重试：远端可能已在第一次连接断开前完成生成，重复提交会带来重复生成或重复计费风险。

### `WebSocket API call timeout`

若日志模块名不是 `astrbot_plugin_image_generation`，该错误不由本插件的 Codex HTTP 请求产生。例如 `[astrbot_plugin_videos_analysis.main:...]` 应在视频分析插件或 AstrBot 的 WebSocket 发送链路中单独排查。

### 返回 `API 错误 (401)` 或 `API 错误 (403)`

确认 API Key 是否正确、是否仍有权限，并确保服务端采用标准 Bearer 认证。

### 返回“未配置 Codex Responses 接口地址”

在 provider 中填写服务根地址。不要只填写路径，也不要填写带 query 的临时链接。

### 返回“响应中未找到图片数据”

确认服务端在同一次响应的 `output[*]` 中返回：

```text
output[*].type == image_generation_call
output[*].result
```

若服务端字段不同，可先保留一段脱敏响应并据此扩展专用适配器；也可以改用 `custom_http` 手工配置请求体和图片提取路径。

### 生成图片 URL 下载失败

优先让服务端返回 `result` 的 Base64 或 data URL。当前适配器为避免把 API Key 发送到未知图片域名，不会在下载图片 URL 时携带生成接口的 Authorization 头。

### 接口返回异步任务或持续事件流

当前 provider 仅处理同步完成响应。异步轮询和 SSE 需要独立支持任务状态、取消、超时与重试，不能用本模板直接接入。
