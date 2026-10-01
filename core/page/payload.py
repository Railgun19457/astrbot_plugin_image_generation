"""Dashboard Page JSON serialization helpers.

These helpers only convert generation task records into JSON-safe payloads and
resolve safe local image paths. Route registration and request handling stay in
``api.py``.
"""

from __future__ import annotations

import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any

from ..tasks.models import GenerationTaskRecord

PAGE_IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
}


class PagePayloadMixin:
    """Serialize Page tasks, images, and uploads into JSON payloads."""

    def _page_datetime(self, value: Any) -> str | None:
        """Serialize a datetime-like value for Page JSON responses."""
        return value.isoformat(timespec="seconds") if value else None

    def _parse_page_datetime(self, value: Any) -> datetime | None:
        """Parse an ISO datetime string from Page query parameters."""
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(
                tzinfo=None
            )
        except ValueError:
            return None

    def _parse_template_fields(
        self,
        preset: str | None,
        preset_label: str | None,
    ) -> dict[str, Any]:
        """Parse stored template summary into preset and persona names.

        Args:
            preset: Stored template summary text (names or labeled summary).
            preset_label: Stored template category label.

        Returns:
            A dict with presets, personas, and display values.
        """
        summary = str(preset or "").strip()
        label = str(preset_label or "").strip()
        presets: list[str] = []
        personas: list[str] = []

        if summary:
            if "预设:" in summary or "人设:" in summary:
                for part in summary.split("；"):
                    chunk = part.strip()
                    if chunk.startswith("预设:"):
                        names = chunk[len("预设:") :].strip()
                        if names:
                            presets.extend(
                                [
                                    name.strip()
                                    for name in names.replace(",", "、").split("、")
                                    if name.strip()
                                ]
                            )
                    elif chunk.startswith("人设:"):
                        names = chunk[len("人设:") :].strip()
                        if names:
                            personas.extend(
                                [
                                    name.strip()
                                    for name in names.replace(",", "、").split("、")
                                    if name.strip()
                                ]
                            )
            elif label in {"预设", "preset"}:
                presets = [
                    name.strip()
                    for name in summary.replace(",", "、").split("、")
                    if name.strip()
                ]
            elif label in {"人设", "persona"}:
                personas = [
                    name.strip()
                    for name in summary.replace(",", "、").split("、")
                    if name.strip()
                ]
            elif label in {"预设/人设", "preset/persona"}:
                # Older mixed labels without structured text: keep as summary only.
                pass
            else:
                # Unknown label: treat the summary as a generic template name list.
                presets = [
                    name.strip()
                    for name in summary.replace(",", "、").split("、")
                    if name.strip()
                ]

        display_parts: list[str] = []
        if presets:
            display_parts.append(f"预设: {'、'.join(presets)}")
        if personas:
            display_parts.append(f"人设: {'、'.join(personas)}")
        display = "；".join(display_parts) if display_parts else summary

        return {
            "presets": presets,
            "personas": personas,
            "template_summary": display or summary,
            "template_label": label
            or (
                "预设/人设"
                if presets and personas
                else ("预设" if presets else ("人设" if personas else ""))
            ),
        }

    def _uploaded_image_path(self, token: str) -> Path | None:
        """Resolve one Page upload token to a safe local path."""
        safe_token = str(token or "").strip()
        if not safe_token or any(part in safe_token for part in ("/", "\\", "..")):
            return None
        path = (self.plugin.page_upload_dir / safe_token).resolve()
        try:
            path.relative_to(self.plugin.page_upload_dir.resolve())
        except ValueError:
            return None
        if not path.is_file():
            return None
        return path

    def _result_image_path(
        self,
        record: GenerationTaskRecord,
        image_index: str | int,
    ) -> Path | None:
        """Resolve one generated image index to a safe result path."""
        try:
            index = int(str(image_index).strip())
        except (TypeError, ValueError):
            return None
        if index < 1 or index > len(record.result_paths):
            return None
        path = Path(record.result_paths[index - 1]).resolve()
        allowed_roots = [
            self.plugin.image_temp_dir.resolve(),
            self.plugin.astrbot_temp_dir.resolve(),
        ]
        for root in allowed_roots:
            try:
                path.relative_to(root)
                if path.is_file():
                    return path
            except ValueError:
                continue
        return None

    def _page_image_mime_type(self, path: Path) -> str:
        """Infer a safe image MIME type from a result path."""
        suffix = path.suffix.lower()
        if suffix in PAGE_IMAGE_MIME_TYPES:
            return PAGE_IMAGE_MIME_TYPES[suffix]
        guessed, _ = mimetypes.guess_type(path.name)
        if guessed and guessed.startswith("image/"):
            return guessed
        return "application/octet-stream"

    def _page_image_payload(
        self,
        record: GenerationTaskRecord,
        index: int,
        path_value: str,
    ) -> dict[str, Any]:
        """Build a safe JSON payload for one generated image entry."""
        filename = Path(path_value).name
        path = self._result_image_path(record, index)
        size = path.stat().st_size if path else 0
        return {
            "index": index,
            "filename": filename,
            "extension": Path(filename).suffix.lower().lstrip("."),
            "mime_type": self._page_image_mime_type(Path(filename)),
            "size": size,
            "available": bool(path),
            "download_endpoint": (
                f"page/tasks/{record.task_id}/images/{index}/download"
            ),
            "preview_endpoint": (f"page/tasks/{record.task_id}/images/{index}/preview"),
        }

    def _page_task_payload(
        self,
        record: GenerationTaskRecord,
        *,
        include_detail: bool = False,
    ) -> dict[str, Any]:
        """Build a safe JSON payload for one generation task record."""
        request_stats = record.request_stats
        result_count = record.result_count or len(record.result_paths)
        finished = request_stats.get("finished", 0)
        total = request_stats.get("total", record.requested_count)
        progress_percent = int((finished / total) * 100) if total else 0
        template_fields = self._parse_template_fields(
            record.preset, record.preset_label
        )
        payload: dict[str, Any] = {
            "task_id": record.task_id,
            "status": record.status.value,
            "status_label": record.status_label,
            "active": record.is_active,
            "source": record.source,
            "unified_msg_origin": str(record.unified_msg_origin or "").strip(),
            "model": record.model or "",
            "message": record.message,
            "error": record.error,
            "prompt_summary": record.prompt_summary,
            "reference_image_count": record.reference_image_count,
            "requested_count": record.requested_count,
            "result_count": result_count,
            "aspect_ratio": record.aspect_ratio,
            "resolution": record.resolution,
            "preset": record.preset or "",
            "preset_label": record.preset_label,
            "presets": template_fields["presets"],
            "personas": template_fields["personas"],
            "template_summary": template_fields["template_summary"],
            "template_label": template_fields["template_label"],
            "created_at": self._page_datetime(record.created_at),
            "started_at": self._page_datetime(record.started_at),
            "finished_at": self._page_datetime(record.finished_at),
            "duration_seconds": record.duration_seconds,
            "queued_seconds": record.queued_seconds,
            "current_index": record.current_index,
            "retry_attempt": record.retry_attempt,
            "max_retry_attempts": record.max_retry_attempts,
            "request_stats": request_stats,
            "progress_percent": max(0, min(progress_percent, 100)),
            "result_images": [
                self._page_image_payload(record, index, path)
                for index, path in enumerate(record.result_paths, 1)
            ],
        }
        if include_detail:
            payload["prompt"] = record.prompt or record.prompt_summary or ""
            payload["items"] = [
                {
                    "index": item.index,
                    "status": item.status.value,
                    "result_count": item.result_count,
                    "error": item.error,
                    "retry_attempts": item.retry_attempts,
                    "max_retry_attempts": item.max_retry_attempts,
                }
                for item in sorted(
                    record.items.values(), key=lambda task_item: task_item.index
                )
            ]
        return payload

    def _gallery_items_for_record(
        self,
        record: GenerationTaskRecord,
    ) -> list[dict[str, Any]]:
        """Build gallery items from one generation task record."""
        generated_at = (
            self._page_datetime(record.finished_at)
            or self._page_datetime(record.started_at)
            or self._page_datetime(record.created_at)
        )
        items: list[dict[str, Any]] = []
        for index, path_value in enumerate(record.result_paths, 1):
            image = self._page_image_payload(record, index, path_value)
            items.append(
                {
                    "id": f"{record.task_id}:{index}",
                    "task_id": record.task_id,
                    "status": record.status.value,
                    "status_label": record.status_label,
                    "source": record.source,
                    "model": record.model or "",
                    "prompt_summary": record.prompt_summary,
                    "preset": record.preset or "",
                    "preset_label": record.preset_label,
                    "aspect_ratio": record.aspect_ratio,
                    "resolution": record.resolution,
                    "created_at": self._page_datetime(record.created_at),
                    "finished_at": self._page_datetime(record.finished_at),
                    "generated_at": generated_at,
                    "image_index": index,
                    **image,
                }
            )
        return items
