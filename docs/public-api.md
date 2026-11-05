# 插件间公共 API

`plugin.public_api` 是供其他 AstrBot 插件调用的 Python API。它复用本插件的生图任务、审核、参考图处理和结果保存流程，但不会主动向用户发送图片；调用方通过任务 ID 或等待接口获取生成图片的本地路径。

公共 API 的返回对象只保留插件间调用常用字段：状态、任务 ID、图片路径、失败原因和少量任务元信息；内部调试字段不对外暴露。

## 获取 API 实例

```python
meta = self.context.get_registered_star("astrbot_plugin_image_generation")
image_plugin = meta.star_cls if meta and meta.activated else None

if not image_plugin:
    return

api = image_plugin.public_api
```

## 注意事项

- 公共 API 只返回任务信息或本地图片路径
- `unified_msg_origin` 可选：
  - 传入时执行会话黑白名单、冷却与群聊/兼容会话额度；同时传入真实 `sender_id` 才能执行个人额度。按实际保存数量结算
  - 不传入时沿用系统调用，不计会话或个人额度；不能携带 `sender_id`
- 提示词审核和图片审核仍按插件配置执行
- `task_id` 由 API 自动生成
- 提交任务会在收集参考图和预约额度前预检查队列容量；队列满时直接返回 `queue_full`
- 任务历史元数据默认会持久化，插件重载后最近的已结束任务仍可查询；重启前未完成的任务会恢复为 `cancelled`，不会继续执行
- 历史持久化内容包括会话标识、提示词摘要、结果路径和错误摘要，可通过“任务持久化”配置关闭或调整保留条数/天数；未结算额度任务仍保存不含提示词和图片路径的最小恢复记录
- 生成的图片默认持久化到插件数据目录的 `generated_images/`，可通过“持久化生成的图片”和“持久化图片数量上限”控制
- 关闭图片持久化或超出数量上限后，历史任务的 `result_paths` 可能指向已被清理的临时文件，调用方需要自行检查路径是否仍存在
- `requested_count` 表示请求图片数量和最大保存数量；`result_count` 表示实际保存的结果图片数量，可能因部分失败少于请求数
- 返回对象的 `code` 来自 `core.public_api.PublicAPIResultCode`，调用方可直接与字符串值比较，也可导入该枚举避免手写返回码

## 调用身份与额度

`submit_generation_task` 与 `generate_image_files` 使用同一身份与额度规则，新增参数均为可选关键字，不要求旧版第三方插件立即修改：

| 调用形态 | 任务记录中的身份 | 额度处理 |
| :--- | :--- | :--- |
| UMO + 真实 sender_id | `user` | 群聊额度（或私聊兼容额度）与个人额度按各自开关共同检查、预留和结算。 |
| 仅 UMO，默认 `auto` | `legacy` | 保留旧会话限额及黑白名单/冷却；跳过无法识别的个人额度，不把 UMO 当个人 ID。 |
| 无 UMO、无 sender_id | `system` | 沿用旧系统调用，不计会话/个人额度；仍执行审核与任务队列限制。 |
| 显式 `system` + UMO，无 sender_id | `system` | 计入群聊/兼容会话额度及会话限制；不计个人额度。 |
| 显式 `user` 缺 sender_id，或 sender_id 缺 UMO | 拒绝 | 返回 `rejected`，`error="invalid_identity"`，不会创建任务或预留。 |

显式 `system` 不能携带 sender_id；`legacy` 携带 sender_id 会按 `user` 处理，不能借此绕过个人额度。非法 `caller_type` 会被拒绝。群私聊通过 AstrBot `MessageSession.from_str()` 和标准 `MessageType` 识别；旧的不透明 UMO 保留会话限额，不误称为群聊。

只有个人额度启用而调用方未传 sender_id 时，兼容旧调用仍然可用，个人额度不会生效。公共 API 是受信任的插件间 Python 接口，不能依靠可省略身份参数阻止恶意插件；面向用户的调用方应传真实事件身份，后台系统任务则明确标注 `caller_type="system"`。`is_admin` 也应来自 `event.is_admin()`，不可由用户文本或 LLM 决定。

任务记录持久化调用身份、发送者、两层额度键、提交日期、预留数量和豁免状态。结果仍通过原 UMO 归属，取消/查询的会话权限不改为 sender_id。等待接口超时只代表调用方停止等待，任务仍执行并保留额度，最终按实际结果结算；需取消时调用取消接口。

## 接口总览

| 接口 | 返回类型 | 说明 |
| :--- | :--- | :--- |
| `submit_generation_task(...)` | `ImageGenerationSubmitResult` | 提交后台任务，立即返回任务 ID。 |
| `get_generation_task(task_id)` | `ImageGenerationTaskSnapshot \| None` | 查询任务快照。 |
| `cancel_generation_task(task_id, *, unified_msg_origin=None)` | `ImageGenerationOperationResult` | 取消任务。 |
| `wait_generation_result(task_id, *, timeout_seconds=None, poll_interval_seconds=0.5)` | `ImageGenerationResult` | 等待任务结束并返回图片路径。 |
| `generate_image_files(...)` | `ImageGenerationResult` | 提交任务并等待完成的快捷方法。 |

## `submit_generation_task(...)`

提交后台生图任务，成功后立即返回任务 ID，不等待图片生成完成。

### 签名

```python
async def submit_generation_task(
    *,
    prompt: str = "",
    unified_msg_origin: str | None = None,
    source: str = "公共接口",
    image_count: int | None = None,
    aspect_ratio: str | None = None,
    resolution: str | None = None,
    reference_image_sources: Any = None,
    reference_image_data: list[ImageData | tuple[bytes, str]] | None = None,
    presets: str | list[str] | None = None,
    personas: str | list[str] | None = None,
    is_admin: bool = False,
    sender_id: str | None = None,
    caller_type: str = "auto",
) -> ImageGenerationSubmitResult
```

### 参数

| 参数 | 说明 |
| :--- | :--- |
| `prompt` | 额外提示词。默认不会从正文匹配模板名称；开启插件“提示词正文匹配”后才会扫描勾选的预设/人设。若 `presets` 或 `personas` 能生成有效提示词，可为空。 |
| `unified_msg_origin` | 可选会话作用域。传入后按该会话检查并扣减额度。 |
| `source` | 任务来源，仅用于日志和任务记录，建议填调用方插件名。 |
| `image_count` | 生成数量；不传使用插件默认值，超过插件配置上限会被截断。 |
| `aspect_ratio` | 宽高比；不传使用插件默认值，如 `"1:1"`、`"16:9"`、`"不指定"`。 |
| `resolution` | 分辨率；不传使用插件默认值，如 `"1K"`、`"1.5K"`、`"2K"`、`"4K"`、`"不指定"`。 |
| `reference_image_sources` | 参考图来源。支持字符串、列表、嵌套列表、包含 `url`/`path`/`file`/`name` 的字典；可为 HTTP(S) URL、本地路径、`file://` URL 或插件数据目录下 `files/...` 路径。本地路径仅允许当前会话 workspace、AstrBot temp 目录和本插件数据目录，并会校验真实图片类型。 |
| `reference_image_data` | 已读取的参考图二进制列表，推荐传入 `ImageData`；仍兼容旧格式 `(image_bytes, mime_type)`。 |
| `presets` | 一个或多个预设名。字符串可用空格分隔多个名称。 |
| `personas` | 一个或多个人设名。字符串可用空格分隔多个名称；人设参考图会在模型支持图生图时自动加入。 |
| `is_admin` | 仅在传入 `unified_msg_origin` 时生效，用于 AstrBot 管理员额度豁免判断，豁免后仍记录用量。 |
| `sender_id` | 可选真实发送者 ID；请直接传 `event.get_sender_id()`。与 UMO 一起用于个人额度，不接受昵称、提示词推断或群 UMO 冒充个人。 |
| `caller_type` | 默认 `"auto"`：根据是否提供 UMO / sender_id 区分用户、旧版会话和系统调用。可显式传 `"user"`、`"legacy"`、`"system"`，见下文。 |

### 行为

- 最终提示词按 `presets` → `personas` → `prompt` 顺序拼接。
- 插件默认不扫描 `prompt` 正文中的模板名称；开启“提示词正文匹配”后才会按勾选项补匹配。
- 预设可以覆盖 `aspect_ratio` 和 `resolution`。
- 参考图会自动下载、读取、大小检查和去重。
- 当前适配器不支持图生图时，所有参考图会被忽略。
- 创建的任务不会主动发送结果，结果路径写入任务快照的 `result_paths`。

### 返回码

| `code` | `ok` | 说明 |
| :--- | :---: | :--- |
| `accepted` | `True` | 任务已提交。 |
| `generator_not_initialized` | `False` | 生成器未初始化。 |
| `api_key_missing` | `False` | 未配置 API Key。 |
| `template_not_found` | `False` | 指定预设或人设不存在。 |
| `empty_prompt` | `False` | 最终提示词为空。 |
| `rate_limited` | `False` | 命中额度、频率或黑名单限制。 |
| `prompt_blocked` | `False` | 提示词审核未通过。 |
| `queue_full` | `False` | 生图任务队列已满。 |
| `task_manager_closed` | `False` | 任务管理器正在关闭或暂不可用。 |
| `task_id_conflict` | `False` | 自动生成的任务 ID 冲突。 |
| `rejected` | `False` | 任务因其他创建错误被拒绝。 |

### 示例

```python
submit = await api.submit_generation_task(
    prompt="未来城市夜景",
    source="astrbot_plugin_example",
    image_count=2,
    aspect_ratio="16:9",
)

if not submit.ok:
    error_message = submit.message
    return

task_id = submit.task_id
```

## `get_generation_task(task_id)`

查询任务快照。任务不存在时返回 `None`。

### 签名

```python
def get_generation_task(task_id: str) -> ImageGenerationTaskSnapshot | None
```

### 示例

```python
snapshot = api.get_generation_task(task_id)

if snapshot and snapshot.status == "succeeded":
    image_paths = snapshot.result_paths
```

## `cancel_generation_task(...)`

取消仍在排队或运行中的任务。

### 签名

```python
def cancel_generation_task(
    task_id: str,
    *,
    unified_msg_origin: str | None = None,
) -> ImageGenerationOperationResult
```

### 参数

| 参数 | 说明 |
| :--- | :--- |
| `task_id` | 要取消的任务 ID。 |
| `unified_msg_origin` | 可选。传入时只允许取消同一会话作用域的任务。 |

### 返回码

| `code` | `ok` | 说明 |
| :--- | :---: | :--- |
| `cancel_requested` | `True` | 已请求取消或任务已取消。 |
| `cancel_failed` | `False` | 任务不存在、已结束或会话不匹配。具体原因看 `message`。 |

### 示例

```python
cancel_result = api.cancel_generation_task(task_id)

if not cancel_result.ok:
    error_message = cancel_result.message
```

## `wait_generation_result(...)`

等待任务结束，并返回生成图片路径。

### 签名

```python
async def wait_generation_result(
    task_id: str,
    *,
    timeout_seconds: float | None = None,
    poll_interval_seconds: float = 0.5,
) -> ImageGenerationResult
```

### 参数

| 参数 | 说明 |
| :--- | :--- |
| `task_id` | 要等待的任务 ID。 |
| `timeout_seconds` | 等待超时时间；`None` 表示不主动超时。 |
| `poll_interval_seconds` | 兼容旧版本保留；当前优先使用任务完成事件等待。 |

`queued`、`running`、`cancelling` 都是等待过程中的中间状态；本接口会继续等待直到任务进入终态、超时或任务不存在。

### 返回码

| `code` | `ok` | 说明 |
| :--- | :---: | :--- |
| `succeeded` | `True` | 任务成功并返回图片路径。 |
| `no_result` | `False` | 任务状态成功，但没有图片路径。 |
| `timeout` | `False` | 等待超时，任务可能仍在后台运行。 |
| `not_found` | `False` | 任务不存在。 |
| `failed` | `False` | 任务失败。 |
| `cancelled` | `False` | 任务已取消。 |

### 示例

```python
result = await api.wait_generation_result(
    task_id,
    timeout_seconds=180,
)

if result.ok:
    image_paths = result.paths
else:
    error_message = result.message
```

## `generate_image_files(...)`

快捷方法：内部先调用 `submit_generation_task()`，提交成功后再调用 `wait_generation_result()`。

### 签名

```python
async def generate_image_files(
    *,
    prompt: str = "",
    unified_msg_origin: str | None = None,
    source: str = "公共接口",
    image_count: int | None = None,
    aspect_ratio: str | None = None,
    resolution: str | None = None,
    reference_image_sources: Any = None,
    reference_image_data: list[ImageData | tuple[bytes, str]] | None = None,
    presets: str | list[str] | None = None,
    personas: str | list[str] | None = None,
    is_admin: bool = False,
    sender_id: str | None = None,
    caller_type: str = "auto",
    timeout_seconds: float | None = None,
    poll_interval_seconds: float = 0.5,
) -> ImageGenerationResult
```

### 参数

除 `timeout_seconds` 和 `poll_interval_seconds` 外，其余生图参数与 `submit_generation_task()` 相同。

| 参数 | 说明 |
| :--- | :--- |
| `timeout_seconds` | 等待任务完成的超时时间；提交失败时不会等待。 |
| `poll_interval_seconds` | 兼容旧版本保留；当前优先使用任务完成事件等待。 |

## 返回码总览

这些返回码由 `core.public_api.PublicAPIResultCode` 定义，所有返回对象的 `code` 字段均使用下表中的字符串值。

| `code` | 适用接口 | `ok` | 说明 |
| :--- | :--- | :---: | :--- |
| `accepted` | `submit_generation_task` | `True` | 任务已提交。 |
| `generator_not_initialized` | `submit_generation_task`、`generate_image_files` | `False` | 生成器未初始化。 |
| `api_key_missing` | `submit_generation_task`、`generate_image_files` | `False` | 当前适配器需要 API Key，但未配置。 |
| `template_not_found` | `submit_generation_task`、`generate_image_files` | `False` | 指定预设或人设不存在。 |
| `empty_prompt` | `submit_generation_task`、`generate_image_files` | `False` | 拼接预设、人设和额外提示词后仍为空。 |
| `rate_limited` | `submit_generation_task`、`generate_image_files` | `False` | 命中黑名单、频率限制或每日额度限制。 |
| `prompt_blocked` | `submit_generation_task`、`generate_image_files` | `False` | 提示词审核未通过。 |
| `queue_full` | `submit_generation_task`、`generate_image_files` | `False` | 生图任务队列已满，任务没有创建。 |
| `task_manager_closed` | `submit_generation_task`、`generate_image_files` | `False` | 任务管理器正在关闭或暂不可用，任务没有创建。 |
| `task_id_conflict` | `submit_generation_task`、`generate_image_files` | `False` | 自动生成的任务 ID 冲突，任务没有创建。 |
| `rejected` | `submit_generation_task`、`generate_image_files` | `False` | 任务因其他创建错误被拒绝。 |
| `cancel_requested` | `cancel_generation_task` | `True` | 已请求取消任务，或任务已处于取消状态。 |
| `cancel_failed` | `cancel_generation_task` | `False` | 任务不存在、任务已结束或会话作用域不匹配。 |
| `not_found` | `wait_generation_result`、`generate_image_files` | `False` | 任务不存在或已被清理。 |
| `timeout` | `wait_generation_result`、`generate_image_files` | `False` | 等待任务完成超时。 |
| `succeeded` | `wait_generation_result`、`generate_image_files` | `True` | 任务成功并返回图片路径。 |
| `no_result` | `wait_generation_result`、`generate_image_files` | `False` | 任务成功结束但没有图片路径。 |
| `failed` | `wait_generation_result`、`generate_image_files` | `False` | 任务失败。 |
| `cancelled` | `wait_generation_result`、`generate_image_files` | `False` | 任务已取消。 |

### 示例：不计用户额度

```python
result = await api.generate_image_files(
    prompt="一只赛博朋克风格的猫",
    source="astrbot_plugin_example",
    image_count=2,
    aspect_ratio="1:1",
    timeout_seconds=180,
)

if result.ok:
    image_paths = result.paths
else:
    error_message = result.message
```

### 示例：同时执行群聊与个人额度

```python
result = await api.generate_image_files(
    prompt="一张像素风头像",
    unified_msg_origin=event.unified_msg_origin,
    is_admin=event.is_admin(),
    sender_id=event.get_sender_id(),
    caller_type="user",
    source="astrbot_plugin_example",
    timeout_seconds=180,
)
```

### 示例：使用预设、人设和参考图

```python
result = await api.generate_image_files(
    presets=["头像", "二次元"],
    personas="看板娘",
    prompt="蓝色眼睛，微笑",
    reference_image_sources=["file:///E:/images/ref.png"],
    source="astrbot_plugin_example",
    timeout_seconds=180,
)
```

### 示例：直接传入图片二进制

```python
with open("E:/images/ref.png", "rb") as f:
    ref_bytes = f.read()

result = await api.generate_image_files(
    prompt="改成像素风",
    reference_image_data=[(ref_bytes, "image/png")],
    source="astrbot_plugin_example",
    timeout_seconds=180,
)
```

## 返回对象

### `ImageGenerationSubmitResult`

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `ok` | `bool` | 是否提交成功。 |
| `code` | `str` | 机器可读状态码。 |
| `message` | `str` | 人类可读说明。 |
| `task_id` | `str \| None` | 提交成功后的任务 ID。 |
| `error` | `str` | 失败原因；成功时为空字符串。 |

### `ImageGenerationResult`

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `ok` | `bool` | 是否成功获得图片路径。 |
| `code` | `str` | 机器可读状态码。 |
| `message` | `str` | 人类可读说明。 |
| `task_id` | `str` | 任务 ID。提交失败且未创建任务时为空字符串。 |
| `paths` | `list[str]` | 生成图片的本地文件路径列表。 |
| `error` | `str` | 失败原因；成功时为空字符串。 |

### `ImageGenerationOperationResult`

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `ok` | `bool` | 操作是否成功。 |
| `code` | `str` | 机器可读状态码。 |
| `message` | `str` | 人类可读说明。 |
| `task_id` | `str \| None` | 目标任务 ID。 |
| `error` | `str` | 失败原因；成功时为空字符串。 |

### `ImageGenerationTaskSnapshot`

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `task_id` | `str` | 任务 ID。 |
| `status` | `str` | 英文状态：`queued`、`running`、`succeeded`、`failed`、`cancelling`、`cancelled`。 |
| `active` | `bool` | 任务是否仍在进行。 |
| `source` | `str` | 任务来源。 |
| `requested_count` | `int` | 请求图片数量与最大保存数量；部分失败时结果数会减少。 |
| `result_count` | `int` | 实际保存的结果图片数量。 |
| `reference_image_count` | `int` | 实际参考图数量。 |
| `aspect_ratio` | `str` | 任务记录中的宽高比参数。 |
| `resolution` | `str` | 任务记录中的分辨率参数。 |
| `result_paths` | `list[str]` | 生成图片路径列表。 |
| `error` | `str` | 失败原因。 |
| `message` | `str` | 当前任务说明。 |
| `created_at` | `datetime` | 创建时间。 |
| `started_at` | `datetime \| None` | 开始运行时间。 |
| `finished_at` | `datetime \| None` | 结束时间。 |
| `duration_seconds` | `float \| None` | 运行耗时；未开始时为 `None`。 |
| `request_stats` | `dict[str, int]` | 子请求统计，包含 `total`、`finished`、`succeeded`、`failed`、`cancelled`、`running`、`pending`、`result_count`。 |
| `items` | `list[dict]` | 子请求列表，每项包含 `index`、`status`、`result_count`、`error`、`retry_attempts`、`max_retry_attempts`。 |
| `finished_request_count` | `int` | 已结束子请求数量。 |
| `running_request_count` | `int` | 运行中子请求数量。 |
| `pending_request_count` | `int` | 等待中子请求数量。 |
| `failed_request_count` | `int` | 失败子请求数量。 |
| `cancelled_request_count` | `int` | 已取消子请求数量。 |

## 完整任务流程示例

```python
submit = await api.submit_generation_task(
    prompt="未来城市夜景",
    source="astrbot_plugin_example",
    reference_image_sources=["https://example.com/ref.png"],
)

if not submit.ok or not submit.task_id:
    error_message = submit.message
    return

snapshot = api.get_generation_task(submit.task_id)

result = await api.wait_generation_result(
    submit.task_id,
    timeout_seconds=180,
)

if result.ok:
    image_paths = result.paths
else:
    error_message = result.message
```
