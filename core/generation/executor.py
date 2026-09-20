from __future__ import annotations

import asyncio
import time

from astrbot.api import logger
from astrbot.api.event import MessageChain
from astrbot.api.star import Context

from ..config.manager import ConfigManager
from ..shared.constants import UNSPECIFIED_OPTION
from ..adapters.generator import ImageGenerator
from .image_processor import ImageProcessor
from ..shared.logging import log_prefix, safe_log_error_body, safe_log_text
from .reference_collector import ensure_image_data
from ..formatting.result import (
    build_result_info_message,
    format_generation_failure_message,
)
from ..audit.safety import SafetyAuditor
from ..tasks.ids import new_task_id
from ..tasks.manager import TaskManager
from ..shared.types import GenerationRequest, ImageCapability, ImageData
from ..tasks.usage import UsageManager
from .image_utils import validate_aspect_ratio, validate_resolution

LOG = log_prefix("GenerationExecutor")


class GenerationExecutor:
    """Execute image generation tasks and delivery side effects."""

    def __init__(
        self,
        *,
        context: Context,
        config_manager: ConfigManager,
        image_processor: ImageProcessor,
        task_manager: TaskManager,
        usage_manager: UsageManager,
        safety_auditor: SafetyAuditor,
    ):
        """Initialize the generation executor.

        Args:
            context: AstrBot runtime context used to send messages.
            config_manager: Runtime plugin configuration manager.
            image_processor: Image persistence and validation helper.
            task_manager: Generation task state manager.
            usage_manager: Usage accounting manager.
            safety_auditor: Prompt and generated image auditor.
        """
        self.context = context
        self.config_manager = config_manager
        self.image_processor = image_processor
        self.task_manager = task_manager
        self.usage_manager = usage_manager
        self.safety_auditor = safety_auditor
        self.generator: ImageGenerator | None = None
        self.request_semaphore: asyncio.Semaphore | None = None

    def update_generator(self, generator: ImageGenerator | None) -> None:
        """Update the active image generator.

        Args:
            generator: Current generator instance, or ``None`` when unavailable.
        """
        self.generator = generator

    def refresh_request_semaphore(self) -> None:
        """Refresh request-level concurrency limit from current configuration."""
        self.request_semaphore = asyncio.Semaphore(
            self.config_manager.max_concurrent_tasks
        )

    async def generate_and_send_image_async(
        self,
        prompt: str,
        unified_msg_origin: str,
        images_data: list[ImageData | tuple[bytes, str]] | None = None,
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
        image_count: int = 1,
        task_id: str | None = None,
        is_usage_limit_admin: bool = False,
        deliver_via_ai: bool = False,
        auto_send: bool = True,
    ) -> None:
        """Generate images and optionally send them to the user.

        Args:
            prompt: Final prompt sent to the image model.
            unified_msg_origin: Target session and usage scope.
            images_data: Optional reference images.
            aspect_ratio: Requested aspect ratio or unspecified marker.
            resolution: Requested resolution or unspecified marker.
            image_count: Requested sub-request count.
            task_id: Existing task ID, or ``None`` for a fallback task ID.
            is_usage_limit_admin: Whether usage limits are bypassed for display.
            deliver_via_ai: Whether results are handed back to the AI tool flow.
            auto_send: Whether to send generated images to the user directly.
        """
        if not self.generator or not self.generator.adapter:
            if task_id:
                self.task_manager.mark_generation_task_failed(
                    task_id, "生图生成器未初始化"
                )
                logger.warning(
                    f"{log_prefix('Task', task_id)} 生成器未初始化，任务提前结束"
                )
            return

        if not task_id:
            task_id = new_task_id()

        capabilities = self.generator.adapter.get_capabilities()
        task_log = log_prefix("Task", task_id)
        logger.debug(
            f"{task_log} 请求尺寸: 宽高比={safe_log_text(aspect_ratio)}，"
            f"分辨率={safe_log_text(resolution)}"
        )
        try:
            image_count = int(image_count)
        except (TypeError, ValueError):
            image_count = self.config_manager.default_image_count
        image_count = max(1, min(image_count, self.config_manager.max_image_count))
        if not (capabilities & ImageCapability.IMAGE_TO_IMAGE) and images_data:
            logger.warning(
                f"{task_log} 当前适配器不支持参考图，已忽略 {len(images_data)} 张图片"
            )
            images_data = None

        if (
            not (capabilities & ImageCapability.ASPECT_RATIO)
            and aspect_ratio != UNSPECIFIED_OPTION
        ):
            logger.debug(
                f"{task_log} 当前适配器不支持指定比例，已忽略参数: {safe_log_text(aspect_ratio)}"
            )
            aspect_ratio = UNSPECIFIED_OPTION

        if (
            not (capabilities & ImageCapability.RESOLUTION)
            and resolution != UNSPECIFIED_OPTION
        ):
            logger.debug(
                f"{task_log} 当前适配器不支持指定分辨率，已忽略参数: {safe_log_text(resolution)}"
            )
            resolution = UNSPECIFIED_OPTION

        final_ar = validate_aspect_ratio(aspect_ratio) or None
        if final_ar == UNSPECIFIED_OPTION:
            final_ar = None
        final_res = validate_resolution(resolution)
        if final_res == UNSPECIFIED_OPTION:
            final_res = None

        images: list[ImageData] = []
        if images_data:
            for image in images_data:
                images.append(ensure_image_data(image))
        self.task_manager.update_generation_task_references(
            task_id,
            reference_image_count=len(images),
        )

        logger.debug(
            f"{task_log} 生图请求已规范化: 数量={image_count}张，参考图={len(images)}张，"
            f"宽高比={safe_log_text(final_ar or UNSPECIFIED_OPTION)}，"
            f"分辨率={safe_log_text(final_res or UNSPECIFIED_OPTION)}"
        )

        try:
            await self._do_generate_and_send(
                prompt,
                unified_msg_origin,
                images,
                final_ar,
                final_res,
                image_count,
                task_id,
                is_usage_limit_admin,
                deliver_via_ai,
                auto_send,
            )
        except (asyncio.CancelledError, Exception):
            self.release_generation_task_quota_once(task_id)
            raise

    def release_generation_task_quota_once(self, task_id: str) -> None:
        """Release reserved quota for one task at most once.

        Args:
            task_id: Generation task ID whose reservation should be released.
        """
        record = self.task_manager.get_generation_task(task_id)
        if (
            not record
            or not record.usage_scope
            or record.quota_released
            or record.quota_settled
        ):
            return
        self.usage_manager.release_reserved_usage(
            record.usage_scope,
            is_admin=record.is_usage_limit_admin,
            count=record.reserved_count,
        )
        self.task_manager.mark_generation_task_quota_released(task_id)

    def settle_generation_task_quota_once(
        self,
        task_id: str,
        *,
        actual_count: int,
    ) -> None:
        """Settle reserved quota for one task at most once.

        Args:
            task_id: Generation task ID whose reservation should be settled.
            actual_count: Number of successfully generated images to record.
        """
        record = self.task_manager.get_generation_task(task_id)
        if not record or not record.usage_scope or record.quota_settled:
            return
        self.usage_manager.settle_usage(
            record.usage_scope,
            is_admin=record.is_usage_limit_admin,
            reserved_count=record.reserved_count,
            actual_count=actual_count,
        )
        self.task_manager.mark_generation_task_quota_settled(task_id)

    async def _do_generate_and_send(
        self,
        prompt: str,
        unified_msg_origin: str,
        images: list[ImageData],
        aspect_ratio: str | None,
        resolution: str | None,
        image_count: int,
        task_id: str,
        is_usage_limit_admin: bool,
        deliver_via_ai: bool = False,
        auto_send: bool = True,
    ) -> None:
        """Execute generation logic and deliver results."""
        start_time = time.time()
        task_log = log_prefix("Task", task_id)
        skip_direct_delivery = deliver_via_ai or not auto_send or not unified_msg_origin
        if not self.generator:
            logger.warning(f"{task_log} 生成器未初始化，跳过生成请求")
            self.task_manager.mark_generation_task_failed(task_id, "生图生成器未初始化")
            return
        logger.debug(
            f"{task_log} 调用生图适配器: 数量={image_count}张，参考图={len(images)}张，"
            f"宽高比={safe_log_text(aspect_ratio or UNSPECIFIED_OPTION)}，"
            f"分辨率={safe_log_text(resolution or UNSPECIFIED_OPTION)}"
        )

        converted_images = await self.generator.convert_reference_images(images)

        generated_file_paths, errors = await self._generate_image_requests_concurrently(
            task_id=task_id,
            task_log=task_log,
            prompt=prompt,
            images=converted_images,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            image_count=image_count,
        )

        duration = time.time() - start_time

        if not generated_file_paths:
            error = "; ".join(errors) or "模型未返回图片"
            self.task_manager.mark_generation_task_failed(task_id, error)
            if skip_direct_delivery:
                return
            await self.context.send_message(
                unified_msg_origin,
                MessageChain().message(format_generation_failure_message(error)),
            )
            return

        logger.debug(
            f"{task_log} 生成请求汇总: 耗时={duration:.2f}秒，"
            f"结果={len(generated_file_paths)}/{image_count}张，失败={len(errors)}项"
        )

        image_allowed, image_reason = await self.safety_auditor.audit_generated_images(
            prompt=prompt,
            image_paths=generated_file_paths,
            unified_msg_origin=unified_msg_origin,
        )
        if not image_allowed:
            self.settle_generation_task_quota_once(
                task_id,
                actual_count=len(generated_file_paths),
            )
            self.task_manager.mark_generation_task_failed(
                task_id,
                f"图片内容审核未通过: {image_reason}",
            )
            if skip_direct_delivery:
                return
            await self.context.send_message(
                unified_msg_origin,
                MessageChain().message(f"❌ 图片内容审核未通过: {image_reason}"),
            )
            return

        result_message = "图片已生成，等待 AI 处理" if deliver_via_ai else "图片已生成"
        if errors:
            result_message = f"{result_message}；部分失败: {'; '.join(errors)}"

        self.settle_generation_task_quota_once(
            task_id,
            actual_count=len(generated_file_paths),
        )

        if skip_direct_delivery:
            self.task_manager.mark_generation_task_succeeded(
                task_id,
                result_count=len(generated_file_paths),
                result_paths=generated_file_paths,
                message=result_message,
            )
            return

        info_message = build_result_info_message(
            self.config_manager,
            self.usage_manager,
            unified_msg_origin=unified_msg_origin,
            is_usage_limit_admin=is_usage_limit_admin,
            duration=duration,
            result_count=len(generated_file_paths),
            task_id=task_id,
        )

        sent_batches, send_errors = await self._send_generated_images(
            unified_msg_origin,
            generated_file_paths,
            info_message=info_message,
        )
        if send_errors:
            delivery_message = (
                f"图片已生成；已发送 {sent_batches} 批，"
                f"发送失败 {len(send_errors)} 批: {'; '.join(send_errors)}"
            )
        else:
            delivery_message = "图片已发送"
        if errors:
            delivery_message = f"{delivery_message}；部分生成失败: {'; '.join(errors)}"

        self.task_manager.mark_generation_task_succeeded(
            task_id,
            result_count=len(generated_file_paths),
            result_paths=generated_file_paths,
            message=delivery_message,
        )

    async def _send_generated_images(
        self,
        unified_msg_origin: str,
        image_paths: list[str],
        *,
        info_message: str = "",
    ) -> tuple[int, list[str]]:
        """Send generated images in configured batches."""
        max_per_message = max(1, self.config_manager.max_images_per_message)
        total = len(image_paths)
        sent_batches = 0
        errors: list[str] = []
        for start in range(0, total, max_per_message):
            batch_paths = image_paths[start : start + max_per_message]
            chain = MessageChain()
            for file_path in batch_paths:
                chain.file_image(file_path)

            is_last_batch = start + max_per_message >= total
            if is_last_batch and info_message:
                chain.message("\n" + info_message)

            batch_index = start // max_per_message + 1
            try:
                await self.context.send_message(unified_msg_origin, chain)
                sent_batches += 1
            except Exception as exc:
                error_message = (
                    f"第 {batch_index} 批发送失败: {safe_log_error_body(exc, 160)}"
                )
                errors.append(error_message)
                logger.error(f"{LOG} {error_message}", exc_info=True)
        return sent_batches, errors

    async def _generate_image_requests_concurrently(
        self,
        *,
        task_id: str,
        task_log: str,
        prompt: str,
        images: list[ImageData],
        aspect_ratio: str | None,
        resolution: str | None,
        image_count: int,
    ) -> tuple[list[str], list[str]]:
        """Generate all requested images concurrently under request-level limits."""
        generated_file_paths: list[str] = []
        errors: list[str] = []
        pending_tasks: dict[asyncio.Task, int] = {}
        next_index = 1
        max_pending_requests = min(
            image_count,
            max(1, self.config_manager.max_concurrent_tasks),
        )

        async def schedule_next_request() -> None:
            nonlocal next_index
            if next_index > image_count:
                return
            current_index = next_index
            next_index += 1
            self.task_manager.update_generation_task_item_result(
                task_id,
                index=current_index,
                status="running",
            )
            self.task_manager.update_generation_task_progress(
                task_id,
                current_index=current_index,
                result_count=len(generated_file_paths),
                message=(
                    f"子请求 {current_index}/{image_count} 已调度，"
                    f"结果图片 {len(generated_file_paths)} 张"
                ),
            )
            task = asyncio.create_task(
                self._generate_one_image_request(
                    GenerationRequest(
                        prompt=prompt,
                        images=images,
                        aspect_ratio=aspect_ratio,
                        resolution=resolution,
                        task_id=task_id,
                        batch_index=current_index,
                        batch_count=image_count,
                        retry_status_callback=lambda retry_attempt,
                        max_retry_attempts,
                        current_index=current_index: (
                            self.task_manager.update_generation_task_retry_status(
                                task_id,
                                current_index=current_index,
                                retry_attempt=retry_attempt,
                                max_retry_attempts=max_retry_attempts,
                            )
                        ),
                    )
                ),
                name=f"image_generation_request:{task_id}:{current_index}",
            )
            pending_tasks[task] = current_index

        while len(pending_tasks) < max_pending_requests:
            await schedule_next_request()

        try:
            while pending_tasks:
                done_tasks, _ = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for done_task in done_tasks:
                    current_index = pending_tasks.pop(done_task)
                    try:
                        result = done_task.result()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        result = None
                        error_message = (
                            f"第 {current_index} 张生成失败: "
                            f"{safe_log_error_body(exc, 200)}"
                        )
                        errors.append(error_message)
                        self.task_manager.update_generation_task_item_result(
                            task_id,
                            index=current_index,
                            status="failed",
                            error=str(exc),
                        )
                        logger.debug(
                            f"{task_log} {safe_log_error_body(error_message, 200)}"
                        )

                    if result is None:
                        continue

                    if result.error:
                        error_message = f"第 {current_index} 张生成失败: {result.error}"
                        errors.append(error_message)
                        self.task_manager.update_generation_task_item_result(
                            task_id,
                            index=current_index,
                            status="failed",
                            error=result.error,
                        )
                        log_failed_item = (
                            logger.warning if image_count > 1 else logger.debug
                        )
                        log_failed_item(
                            f"{task_log} {safe_log_error_body(error_message, 200)}"
                        )
                    elif not result.images:
                        error_message = f"第 {current_index} 张生成失败: 模型未返回图片"
                        errors.append(error_message)
                        self.task_manager.update_generation_task_item_result(
                            task_id,
                            index=current_index,
                            status="failed",
                            error="模型未返回图片",
                        )
                        log_failed_item = (
                            logger.warning if image_count > 1 else logger.debug
                        )
                        log_failed_item(f"{task_log} {error_message}")
                    else:
                        saved_count = 0
                        for img_bytes in result.images:
                            file_path = self.image_processor.save_generated_image(
                                task_id, img_bytes
                            )
                            if file_path:
                                generated_file_paths.append(file_path)
                                saved_count += 1
                            else:
                                error_message = (
                                    f"第 {current_index} 张生成失败: 未能保存图片"
                                )
                                errors.append(error_message)
                                self.task_manager.update_generation_task_item_result(
                                    task_id,
                                    index=current_index,
                                    status="failed",
                                    result_count=saved_count,
                                    error="未能保存图片",
                                )
                                log_failed_item = (
                                    logger.warning if image_count > 1 else logger.debug
                                )
                                log_failed_item(f"{task_log} {error_message}")
                                break
                        else:
                            self.task_manager.update_generation_task_item_result(
                                task_id,
                                index=current_index,
                                status="succeeded",
                                result_count=saved_count,
                            )

                    self.task_manager.update_generation_task_progress(
                        task_id,
                        current_index=min(next_index, image_count),
                        result_count=len(generated_file_paths),
                        message=(
                            f"子请求 {min(next_index - 1, image_count)}/{image_count} 已完成，"
                            f"结果图片 {len(generated_file_paths)} 张"
                        ),
                    )

                    if next_index <= image_count:
                        await schedule_next_request()
        except asyncio.CancelledError:
            self.task_manager.mark_unfinished_generation_task_items_cancelled(
                task_id,
                reason="任务已取消",
            )
            raise
        finally:
            for pending_task in pending_tasks:
                pending_task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)

        return generated_file_paths, errors

    async def _generate_one_image_request(
        self,
        request: GenerationRequest,
    ):
        """Run one adapter generation request under the request-level semaphore."""
        if not self.generator:
            return None
        if self.request_semaphore is None:
            return await self.generator.generate_preconverted(
                request,
                images=request.images,
            )
        async with self.request_semaphore:
            return await self.generator.generate_preconverted(
                request,
                images=request.images,
            )
