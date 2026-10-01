"""Backend API handlers for the plugin Dashboard Page."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from astrbot.api.web import (
    PluginUploadFile,
    error_response,
    file_response,
    json_response,
    request,
)

from ..shared.logging import safe_log_text
from ..shared.types import ImageCapability
from ..tasks.models import GenerationTaskRecord
from .payload import PagePayloadMixin
from .stats import PageStatsMixin

PLUGIN_NAME = "astrbot_plugin_image_generation"
PAGE_PREVIEW_MAX_BYTES = 12 * 1024 * 1024
PAGE_IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/heic": ".heic",
    "image/heif": ".heif",
}


class ImageGenerationPageAPI(PagePayloadMixin, PageStatsMixin):
    """Register and serve backend endpoints used by the Dashboard Page."""

    def __init__(self, plugin: Any) -> None:
        self.plugin = plugin

    def register(self) -> None:
        """Register backend APIs used by the plugin Dashboard Page."""
        page_routes = [
            ("/page/state", self.page_state, ["GET"], "Image generation Page state"),
            (
                "/page/stats",
                self.page_stats,
                ["GET"],
                "Image generation Page statistics",
            ),
            ("/page/tasks", self.page_tasks, ["GET"], "Image generation Page tasks"),
            (
                "/page/tasks/<task_id>",
                self.page_task_detail,
                ["GET"],
                "Image generation Page task detail",
            ),
            (
                "/page/tasks/<task_id>/cancel",
                self.page_cancel_task,
                ["POST"],
                "Cancel image generation Page task",
            ),
            (
                "/page/tasks/<task_id>/images/<image_index>/download",
                self.page_download_task_image,
                ["GET"],
                "Download generated image from Page",
            ),
            (
                "/page/tasks/<task_id>/images/<image_index>/preview",
                self.page_preview_task_image,
                ["GET"],
                "Preview generated image from Page",
            ),
            (
                "/page/gallery",
                self.page_gallery,
                ["GET"],
                "Image generation Page gallery",
            ),
            (
                "/page/reference/upload",
                self.page_upload_reference,
                ["POST"],
                "Upload reference image from Page",
            ),
            (
                "/page/generate",
                self.page_generate,
                ["POST"],
                "Submit image generation task from Page",
            ),
        ]
        for route, handler, methods, description in page_routes:
            self.plugin.context.register_web_api(
                f"/{PLUGIN_NAME}{route}",
                handler,
                methods,
                description,
            )

    async def page_state(self):
        """Return plugin state used to initialize the Dashboard Page.

        Returns:
            A JSON response with model, template, capability, and runtime state.
        """
        plugin = self.plugin
        adapter_config = plugin.config_manager.adapter_config
        capabilities = (
            plugin.generator.adapter.get_capabilities()
            if plugin.generator and plugin.generator.adapter
            else ImageCapability.NONE
        )
        current_model = (
            f"{adapter_config.name}/{adapter_config.model}" if adapter_config else ""
        )
        available_models: list[str] = []
        for provider in getattr(plugin.config_manager, "_all_provider_configs", []):
            provider_name = provider.name.strip()
            for model in provider.available_models:
                available_models.append(
                    f"{provider_name}/{model}" if provider_name else model
                )
        return json_response(
            {
                "initialized": bool(plugin.generator and plugin.generator.adapter),
                "has_api_key": plugin.has_required_api_key(),
                "current_model": current_model,
                "available_models": available_models,
                "default_image_count": plugin.config_manager.default_image_count,
                "max_image_count": plugin.config_manager.max_image_count,
                "default_aspect_ratio": plugin.config_manager.default_aspect_ratio,
                "default_resolution": plugin.config_manager.default_resolution,
                "supports_image_to_image": bool(
                    capabilities & ImageCapability.IMAGE_TO_IMAGE
                ),
                "supports_aspect_ratio": bool(
                    capabilities & ImageCapability.ASPECT_RATIO
                ),
                "supports_resolution": bool(capabilities & ImageCapability.RESOLUTION),
                "presets": [
                    {"name": name, "summary": safe_log_text(value, 120)}
                    for name, value in plugin.config_manager.presets.items()
                ],
                "personas": [
                    {
                        "name": name,
                        "summary": safe_log_text(persona.prompt, 120),
                        "has_image": bool(persona.image),
                    }
                    for name, persona in plugin.config_manager.personas.items()
                ],
                "queue": {
                    "can_accept": plugin.task_manager.can_accept_generation_task(),
                    "max_running": plugin.config_manager.max_running_generation_tasks,
                    "max_queued": plugin.config_manager.max_queued_generation_tasks,
                },
                "history": {
                    "enabled": plugin.config_manager.enable_generation_task_history,
                    "limit": plugin.config_manager.generation_task_history_limit,
                    "retention_days": plugin.config_manager.generation_task_history_retention_days,
                },
            }
        )

    async def page_stats(self):
        """Return aggregated dashboard statistics for the overview page.

        Returns:
            A JSON response with summary cards and chart-ready distributions.
        """
        days_raw = str(request.query.get("days", "7") or "7").strip()
        try:
            days = max(0, int(days_raw))
        except ValueError:
            return error_response("days 必须是非负整数", status_code=400)
        return json_response(self._page_stats_payload(days=days))

    async def page_tasks(self):
        """Return generation tasks for the Dashboard Page.

        Returns:
            A JSON response containing filtered and paginated task summaries.
        """
        status_filter = str(request.query.get("status", "all") or "all").strip()
        source_filter = str(request.query.get("source", "") or "").strip().lower()
        model_filter = str(request.query.get("model", "") or "").strip().lower()
        keyword = str(request.query.get("keyword", "") or "").strip().lower()
        days_raw = str(request.query.get("days", "0") or "0").strip()
        try:
            days = max(0, int(days_raw))
        except ValueError:
            return error_response("days 必须是非负整数", status_code=400)
        limit = max(1, min(request.query.get("limit", 30, type=int), 200))
        offset = max(0, request.query.get("offset", 0, type=int))
        since = self._parse_page_datetime(request.query.get("since"))
        until = self._parse_page_datetime(request.query.get("until"))
        if days > 0 and since is None:
            since = datetime.now() - timedelta(days=days)

        # Scan full local history so the queue can page through all saved tasks.
        scan_limit = min(
            5000,
            max(
                self.plugin.config_manager.generation_task_history_limit,
                limit + offset,
                200,
            ),
        )
        records = self.plugin.task_manager.list_generation_tasks(
            include_finished=True,
            limit=scan_limit,
        )

        filtered: list[GenerationTaskRecord] = []
        model_names: set[str] = set()
        source_names: set[str] = set()
        for record in records:
            record_model = str(record.model or "").strip()
            record_source = str(record.source or "").strip()
            if record_model:
                model_names.add(record_model)
            if record_source:
                source_names.add(record_source)

            if status_filter == "active":
                if not record.is_active:
                    continue
            elif status_filter not in {"", "all"}:
                if record.status.value != status_filter:
                    continue

            if source_filter and source_filter not in record_source.lower():
                continue
            if model_filter and model_filter not in record_model.lower():
                continue

            record_time = record.finished_at or record.started_at or record.created_at
            if since and record_time and record_time < since:
                continue
            if until and record_time and record_time > until:
                continue

            if keyword:
                haystack = " ".join(
                    [
                        record.task_id,
                        record.prompt_summary or "",
                        record.prompt or "",
                        record.preset or "",
                        record.preset_label or "",
                        record.source or "",
                        record.model or "",
                        record.status.value,
                        record.unified_msg_origin or "",
                        record.message or "",
                        record.error or "",
                    ]
                ).lower()
                if keyword not in haystack:
                    continue
            filtered.append(record)

        total = len(filtered)
        page_records = filtered[offset : offset + limit]
        return json_response(
            {
                "tasks": [self._page_task_payload(record) for record in page_records],
                "total": total,
                "offset": offset,
                "limit": limit,
                "models": sorted(model_names),
                "sources": sorted(source_names),
            }
        )

    async def page_task_detail(self, task_id: str):
        """Return one generation task detail for the Dashboard Page.

        Args:
            task_id: Generation task ID from the route.

        Returns:
            A JSON response with task detail or an error response.
        """
        normalized_task_id = str(task_id or "").strip()
        record = self.plugin.task_manager.get_generation_task(normalized_task_id)
        if not record:
            return error_response(f"任务不存在: {normalized_task_id}", status_code=404)
        return json_response(
            {"task": self._page_task_payload(record, include_detail=True)}
        )

    async def page_cancel_task(self, task_id: str):
        """Cancel an active generation task from the Dashboard Page.

        Args:
            task_id: Generation task ID from the route.

        Returns:
            A JSON response with operation status.
        """
        normalized_task_id = str(task_id or "").strip()
        ok, message = self.plugin.task_manager.cancel_generation_task(
            normalized_task_id
        )
        status_code = 200 if ok else 400
        return json_response(
            {"ok": ok, "message": message, "task_id": normalized_task_id},
            status_code=status_code,
        )

    async def page_download_task_image(self, task_id: str, image_index: str):
        """Download one generated image file through a safe Page endpoint.

        Args:
            task_id: Generation task ID from the route.
            image_index: 1-based image index in the task result list.

        Returns:
            A file response or an error response.
        """
        normalized_task_id = str(task_id or "").strip()
        record = self.plugin.task_manager.get_generation_task(normalized_task_id)
        if not record:
            return error_response(f"任务不存在: {normalized_task_id}", status_code=404)
        path = self._result_image_path(record, image_index)
        if not path:
            return error_response("图片不存在或不可访问", status_code=404)
        return file_response(path, filename=path.name)

    async def page_preview_task_image(self, task_id: str, image_index: str):
        """Return a small inline preview payload for one generated image.

        Args:
            task_id: Generation task ID from the route.
            image_index: 1-based image index in the task result list.

        Returns:
            A JSON response with a data URL preview, or an error response.
        """
        normalized_task_id = str(task_id or "").strip()
        record = self.plugin.task_manager.get_generation_task(normalized_task_id)
        if not record:
            return error_response(f"任务不存在: {normalized_task_id}", status_code=404)
        path = self._result_image_path(record, image_index)
        if not path:
            return error_response("图片不存在或不可访问", status_code=404)
        size = path.stat().st_size
        if size > PAGE_PREVIEW_MAX_BYTES:
            return error_response(
                "图片过大，请改用下载查看",
                status_code=413,
                data={
                    "download_endpoint": (
                        f"page/tasks/{normalized_task_id}/images/{image_index}/download"
                    ),
                    "size": size,
                },
            )
        mime_type = self._page_image_mime_type(path)
        data = path.read_bytes()
        return json_response(
            {
                "task_id": normalized_task_id,
                "image_index": int(str(image_index).strip()),
                "filename": path.name,
                "mime_type": mime_type,
                "size": size,
                "data_url": (
                    f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"
                ),
            }
        )

    async def page_gallery(self):
        """Return generated image gallery items from task history.

        Returns:
            A JSON response containing filtered gallery image entries.
        """
        status_filter = str(request.query.get("status", "all") or "all").strip()
        model_filter = str(request.query.get("model", "") or "").strip().lower()
        keyword = str(request.query.get("keyword", "") or "").strip().lower()
        days_raw = str(request.query.get("days", "0") or "0").strip()
        try:
            days = max(0, int(days_raw))
        except ValueError:
            return error_response("days 必须是非负整数", status_code=400)
        limit = max(1, min(request.query.get("limit", 60, type=int), 200))
        offset = max(0, request.query.get("offset", 0, type=int))
        since = self._parse_page_datetime(request.query.get("since"))
        until = self._parse_page_datetime(request.query.get("until"))
        if days > 0 and since is None:
            since = datetime.now() - timedelta(days=days)

        # Scan more task records than page size because one task can yield
        # multiple gallery images.
        scan_limit = min(1000, max((limit + offset) * 5, 200))
        records = self.plugin.task_manager.list_generation_tasks(
            include_finished=True,
            limit=scan_limit,
        )
        items: list[dict[str, Any]] = []
        model_names: set[str] = set()
        for record in records:
            record_model = str(record.model or "").strip()
            if record_model:
                model_names.add(record_model)
            if (
                status_filter not in {"", "all"}
                and record.status.value != status_filter
            ):
                continue
            if model_filter and model_filter not in record_model.lower():
                continue
            record_time = record.finished_at or record.started_at or record.created_at
            if since and record_time and record_time < since:
                continue
            if until and record_time and record_time > until:
                continue
            if keyword:
                haystack = " ".join(
                    [
                        record.task_id,
                        record.prompt_summary or "",
                        record.preset or "",
                        record.source or "",
                        record.model or "",
                        record.status.value,
                    ]
                ).lower()
                if keyword not in haystack:
                    continue
            items.extend(self._gallery_items_for_record(record))

        total = len(items)
        page_items = items[offset : offset + limit]
        return json_response(
            {
                "items": page_items,
                "total": total,
                "offset": offset,
                "limit": limit,
                "available_count": sum(
                    1 for item in page_items if item.get("available")
                ),
                "models": sorted(model_names),
            }
        )

    async def page_upload_reference(self):
        """Upload and validate one reference image for Page generation.

        Returns:
            A JSON response containing an opaque upload token.
        """
        files = await request.files()
        upload = files.get("file")
        if not isinstance(upload, PluginUploadFile):
            return error_response("缺少上传文件", status_code=400)
        data = await upload.read()
        max_bytes = (
            self.plugin.config_manager.usage_settings.max_image_size_mb * 1024 * 1024
        )
        if not data:
            return error_response("上传文件为空", status_code=400)
        if len(data) > max_bytes:
            return error_response(
                f"参考图超过大小限制 ({self.plugin.config_manager.usage_settings.max_image_size_mb}MB)",
                status_code=400,
            )
        image_data = self.plugin.image_processor.validate_image_data(
            data,
            log_source=upload.filename or "Page upload",
        )
        if not image_data:
            return error_response("上传文件不是受支持的图片", status_code=400)
        suffix = PAGE_IMAGE_EXTENSIONS.get(image_data.mime_type, ".png")
        digest = hashlib.md5(data).hexdigest()[:12]
        token = (
            f"upload_{int(asyncio.get_running_loop().time() * 1000)}_{digest}{suffix}"
        )
        target = (self.plugin.page_upload_dir / token).resolve()
        try:
            target.relative_to(self.plugin.page_upload_dir.resolve())
        except ValueError:
            return error_response("上传路径无效", status_code=400)
        with target.open("wb") as output:
            output.write(data)
        return json_response(
            {
                "token": token,
                "filename": Path(upload.filename or token).name,
                "mime_type": image_data.mime_type,
                "size": len(data),
            }
        )

    async def page_generate(self):
        """Submit one image generation task from the Dashboard Page.

        Returns:
            A JSON response with submitted task information.
        """
        plugin = self.plugin
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            return error_response("请求体必须是 JSON 对象", status_code=400)
        prompt = str(payload.get("prompt") or "").strip()
        image_count = plugin.normalize_image_count(payload.get("image_count"))
        aspect_ratio = str(
            payload.get("aspect_ratio") or plugin.config_manager.default_aspect_ratio
        ).strip()
        resolution = str(
            payload.get("resolution") or plugin.config_manager.default_resolution
        ).strip()
        model = str(payload.get("model") or "").strip()
        presets = payload.get("presets") or None
        personas = payload.get("personas") or None
        upload_tokens = payload.get("reference_tokens") or []
        reference_sources = [
            str(path)
            for token in upload_tokens
            if (path := self._uploaded_image_path(str(token)))
        ]
        if model and model != (
            f"{plugin.config_manager.adapter_config.name}/{plugin.config_manager.adapter_config.model}"
            if plugin.config_manager.adapter_config
            else ""
        ):
            plugin.config_manager.save_model_setting(model)
            plugin.config_manager.reload()
            plugin.reload_runtime_settings()
            if plugin.generator and plugin.config_manager.adapter_config:
                await plugin.generator.update_adapter(
                    plugin.config_manager.adapter_config
                )
                plugin.generation_executor.update_generator(plugin.generator)

        # Page generation runs from Dashboard auth context. Prefer the signed-in
        # WebUI username so overview statistics can attribute tasks correctly.
        webui_username = str(getattr(request, "username", None) or "").strip()
        page_user_origin = (
            f"webui:{webui_username}" if webui_username else "webui:admin"
        )

        result = await plugin.public_api.submit_generation_task(
            prompt=prompt,
            source="Page",
            unified_msg_origin=page_user_origin,
            image_count=image_count,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            reference_image_sources=reference_sources,
            presets=presets,
            personas=personas,
            is_admin=True,
        )
        if not result.ok:
            return error_response(
                result.message,
                status_code=400,
                data={"code": result.code, "error": result.error},
            )
        record = plugin.task_manager.get_generation_task(result.task_id or "")
        return json_response(
            {
                "ok": True,
                "message": result.message,
                "task_id": result.task_id,
                "task": self._page_task_payload(record) if record else None,
            }
        )
