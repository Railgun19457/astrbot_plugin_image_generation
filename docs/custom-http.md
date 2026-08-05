# 自定义 HTTP 接口配置(Beta)

`custom_http` 适配器用于接入插件未内置的图像生成 HTTP 接口。它通过配置模板构造请求，并从响应中按路径提取图片数据。

当前版本主要支持：

- JSON 请求体接口
- JSON 响应中的 base64 图片、data URL 或 HTTP 图片 URL
- 直接返回 `image/*` 二进制图片的接口
- 文生图和通过 JSON 字段传入参考图的图生图

暂不支持：

- `multipart/form-data` 文件上传接口
- 提交任务后需要轮询结果的异步接口
- 非 JSON 的复杂响应解析

## 快速配置

1. 在“图像模型供应商”中新增 **自定义 HTTP 接口**。
2. 填写 `供应商名称`、`API Base URL`、`Endpoint`、`可用模型列表`。
3. 按接口文档维护 `请求头 JSON`、`查询参数 JSON`、`请求体 JSON`。
4. 设置 `图片结果路径` 和 `图片结果类型`。
5. 在“生图模型”中选择 `供应商名称/模型名称`。

JSON 类型配置项已启用代码编辑器模式，适合编辑复杂 payload。

## URL 构造规则

`custom_http` 使用 `API Base URL` 和 `Endpoint` 构造请求地址：

- 如果 `Endpoint` 是完整 `http://` 或 `https://` 地址，则直接使用 `Endpoint`。
- 否则使用 `API Base URL + Endpoint`。
- 自定义 HTTP 不会自动移除 `API Base URL` 中的 `/v1` 路径。

示例：

| API Base URL | Endpoint | 实际请求地址 |
| :--- | :--- | :--- |
| `https://api.example.com` | `/v1/images/generations` | `https://api.example.com/v1/images/generations` |
| `https://api.example.com/v1` | `/images/generations` | `https://api.example.com/v1/images/generations` |
| 留空 | `https://api.example.com/v1/images` | `https://api.example.com/v1/images` |

## 配置项说明

| 配置项 | 说明 |
| :--- | :--- |
| `供应商名称` | 模型选择前缀，最终格式为 `供应商名称/模型名称`。 |
| `API Base URL` | 接口基础地址。可包含 `/v1` 等路径。 |
| `Endpoint` | 接口路径或完整 URL。支持占位符。 |
| `请求方法` | 支持 `GET`、`POST`、`PUT`、`PATCH`、`DELETE`；推荐 `POST`。 |
| `代理地址` | 可选 HTTP 代理，例如 `http://127.0.0.1:7890`。 |
| `API 密钥` | 可选；模板中用 `{api_key}` 引用。多个 Key 会随重试轮换。 |
| `可用模型列表` | `/生图模型` 中展示的模型列表；模板中用 `{model}` 引用。 |
| `模型能力` | 按接口实际能力勾选文生图、图生图、宽高比、分辨率。 |
| `请求头 JSON` | JSON 对象，作为 HTTP Header。 |
| `查询参数 JSON` | JSON 对象，作为 URL query params。 |
| `请求体 JSON` | JSON 对象，作为 JSON payload。`GET` 和 `DELETE` 不发送请求体。 |
| `图片结果路径` | 从 JSON 响应中提取图片数据的路径。多个路径可用换行或逗号分隔。 |
| `图片结果类型` | `auto`、`base64`、`data_url`、`url`。推荐使用 `auto`。 |
| `错误信息路径` | 从 JSON 响应中提取错误信息的路径。留空时尝试 `error.message` 和 `error`。 |
| `成功 HTTP 状态码` | 响应状态码在列表中时才会继续解析响应。默认 `[200]`。 |

## 占位符

请求头、查询参数、请求体和 Endpoint 都支持占位符。

| 占位符 | 含义 |
| :--- | :--- |
| `{prompt}` | 最终提示词。 |
| `{model}` | 当前模型名称。 |
| `{api_key}` | 当前使用的 API Key。 |
| `{aspect_ratio}` | 宽高比；未指定时为空字符串。 |
| `{resolution}` | 分辨率；未指定时为空字符串。 |
| `{task_id}` | 当前任务 ID。 |
| `{batch_index}` | 当前子请求序号。 |
| `{batch_count}` | 当前任务总子请求数。 |
| `{requested_count}` | 请求生成数量，同 `{batch_count}`。 |
| `{image_count}` / `{count}` | 当前单次接口请求出图数，固定为 `1`。 |
| `{reference_image_count}` | 参考图数量。 |
| `{reference_images}` | 参考图 data URL 数组。 |
| `{reference_images_data_url}` | 参考图 data URL 数组。 |
| `{reference_images_data_url_nullable}` | 参考图 data URL 数组，可空。 |
| `{reference_images_base64}` | 参考图纯 base64 数组。 |
| `{reference_images_base64_nullable}` | 参考图纯 base64 数组，可空。 |
| `{reference_images_mime_types}` | 参考图 MIME 类型数组。 |
| `{reference_image_0}` | 第一张参考图 data URL。 |
| `{reference_image_0_data_url}` | 第一张参考图 data URL。 |
| `{reference_image_0_base64}` | 第一张参考图纯 base64。 |
| `{reference_image_0_mime_type}` | 第一张参考图 MIME 类型。 |

### 占位符类型规则

如果 JSON 字段值**完全等于**某个占位符，且该占位符对应数组或数字，会按原类型写入：

```json
{
  "images": "{reference_images_data_url}",
  "n": "{image_count}"
}
```

渲染后等价于：

```json
{
  "images": ["data:image/png;base64,..."],
  "n": 1
}
```

**注意**：在此情形下，表中以`_nullable`结尾的两个占位符，当原数组为空数组时渲染后json中将不带此项，否则与本身值相同：

```json
{
  "images": "{reference_images_base64_nullable}",
  "prompt": "{prompt}"
}
```

当没有参考图时，将被渲染为：

```json
{
  "prompt": "Draw a cat."
}
```

如果占位符只是字符串的一部分，则仍然会按字符串替换：

```json
{
  "prompt": "Generate an image: {prompt}"
}
```

## 响应图片提取路径

`图片结果路径` 支持轻量点号路径语法：

- `data.0.b64_json`：取 `data` 数组第 1 项的 `b64_json`
- `data.*.b64_json`：取 `data` 数组每一项的 `b64_json`
- `data.*.url`：取 `data` 数组每一项的 `url`
- `$`：使用整个响应作为图片数据

支持的图片值：

- 纯 base64 字符串
- `data:image/png;base64,...` 形式的 data URL
- `http://` 或 `https://` 图片 URL
- 对象中的 `b64_json`、`base64`、`image`、`data`、`url`、`image_url.url`

多个路径可以用换行或逗号分隔：

```text
data.*.b64_json
choices.0.message.images.*.url
```

## 示例：OpenAI Images 风格接口

适合返回 `data[].b64_json` 的接口。

### 基础配置

| 配置项 | 示例 |
| :--- | :--- |
| API Base URL | `https://api.example.com` |
| Endpoint | `/v1/images/generations` |
| 请求方法 | `POST` |
| 图片结果路径 | `data.*.b64_json` |
| 图片结果类型 | `auto` |

### 请求头 JSON

```json
{
  "Authorization": "Bearer {api_key}",
  "Content-Type": "application/json"
}
```

如果未配置 API Key，空的 `Bearer` 请求头会被自动忽略。

### 请求体 JSON

```json
{
  "model": "{model}",
  "prompt": "{prompt}",
  "n": 1,
  "response_format": "b64_json"
}
```

## 示例：返回图片 URL 的接口

适合返回 `data[].url` 的接口。

### 请求体 JSON

```json
{
  "model": "{model}",
  "prompt": "{prompt}"
}
```

### 图片结果路径

```text
data.*.url
```

插件会自动下载这些 URL 对应的图片。

## 示例：传入参考图数组

适合图生图接口使用 JSON 字段接收 data URL 数组。

```json
{
  "model": "{model}",
  "prompt": "{prompt}",
  "images": "{reference_images_data_url}"
}
```

如果接口只接收第一张参考图的纯 base64：

```json
{
  "model": "{model}",
  "prompt": "{prompt}",
  "image": "{reference_image_0_base64}"
}
```

## 常见问题

### 提示 `payload_json JSON 解析失败`

请求体不是合法 JSON。注意 JSON 字符串必须使用双引号，不能有尾随逗号。

### 提示 `响应中未找到图片数据`

通常是 `图片结果路径` 与接口响应结构不一致。先查看接口文档或日志中的响应摘要，再调整路径，例如从 `data.*.b64_json` 改为 `images.*.url`。

### 接口返回 200 但插件当作失败

检查 `错误信息路径` 是否误指向普通消息字段。留空时只会尝试 `error.message` 和 `error`。

### 接口成功状态码不是 200

将对应状态码加入 `成功 HTTP 状态码`，例如 `[200, 201, 202]`。

### 需要 multipart 或任务轮询接口

当前 `custom_http` 第一版只覆盖同步 JSON 接口。multipart 和异步轮询接口需要后续扩展独立支持。