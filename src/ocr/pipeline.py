"""Оркестратор OCR-пайплайна: разбивка → рендер → OCR → текстовый слой → сборка."""
import asyncio
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiohttp
from aiohttp import ClientTimeout, TCPConnector

from logging_config import get_logger

from . import overlay, pdf_merge, pdf_split, render
from .client import OcrClient
from .fsutil import cleanup_dir, read_bytes, remove_file, write_bytes
from .layout import PathLayout
from .outputs import ResultWriter
from .ratelimit import AsyncRateLimiter, OCRAuthError

logger = get_logger("ocr")


class YandexOCRAsync:
    def __init__(
        self,
        api_key: str,
        folder_id: str,
        storage_root: Path,
        key: str,
        tmp_prefix: str = "OCR-tmp",
        result_prefix: str = "OCR-result",
        dpi: int = 200,
        jpeg_quality: int = 90,
        strict_memory_mode: bool = True,
        submit_rps: float = 8.0,
        poll_rps: float = 40.0,
        max_batch_bytes: int = 9_000_000,
        max_failed_pages: int = 0,
    ):
        self.key = key
        self.base_name = Path(key).stem
        self.dpi = dpi
        self.jpeg_quality = jpeg_quality
        self.strict_memory_mode = strict_memory_mode
        self.max_batch_bytes = max_batch_bytes
        self.max_failed_pages = max_failed_pages

        self.layout = PathLayout(storage_root, self.base_name, tmp_prefix, result_prefix)
        self.result_writer = ResultWriter(self.layout)

        headers = {"Authorization": f"Api-Key {api_key}", "x-folder-id": folder_id}
        self.client = OcrClient(
            headers=headers,
            submit_limiter=AsyncRateLimiter(submit_rps),
            poll_limiter=AsyncRateLimiter(poll_rps),
        )

        # Шрифт регистрируется один раз (до тредов) — дальше read-only из CPU-воркера.
        self.font_name = overlay.register_font()

        # Один CPU-воркер: fitz (PyMuPDF) не потокобезопасен между потоками, поэтому
        # рендер/overlay сериализуются, но НЕ блокируют event loop (сеть OCR идёт параллельно).
        self._cpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-cpu")

        # Учёт распознанных/потерянных страниц за прогон (для статуса failed).
        self._pages_ok = 0
        self._pages_failed = 0
        self._failed_pages: list[int] = []

        self.layout.ensure_dirs()

    async def _run_cpu(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._cpu, lambda: func(*args))

    def cleanup_tmp_files(self) -> None:
        cleanup_dir(self.layout.tmp_dir)

    def _mark_pages_failed(self, page_numbers: list[int]) -> None:
        self._pages_failed += len(page_numbers)
        self._failed_pages.extend(p + 1 for p in page_numbers)

    def _save_empty_pages(self, chunk_id: int, page_numbers: list[int]) -> None:
        for page_num in page_numbers:
            try:
                self.result_writer.save_page_text(chunk_id, page_num, "")
                self.result_writer.save_page_json(chunk_id, page_num, "")
            except Exception as save_exc:
                logger.warning(f"Failed to save empty result for page {page_num + 1}: {save_exc}")

    async def _process_subbatch(
        self,
        session: aiohttp.ClientSession,
        chunk_id: int,
        batch_id: int,
        image_pdf_path: Path,
        sub_pages: list[int],
        per_page_paths: list[Path],
    ) -> None:
        try:
            image_bytes = read_bytes(image_pdf_path)
            try:
                operation_id = await self.client.recognize_pdf(session, image_bytes)
            finally:
                del image_bytes
                gc.collect()

            if not operation_id:
                logger.error(f"OCR start failed for chunk {chunk_id}, batch {batch_id}")
                self._mark_pages_failed(sub_pages)
                self._save_empty_pages(chunk_id, sub_pages)
                return

            ocr_result = await self.client.get_operation_result(session, operation_id)
            if not ocr_result:
                logger.error(f"OCR result failed for chunk {chunk_id}, batch {batch_id}")
                self._mark_pages_failed(sub_pages)
                self._save_empty_pages(chunk_id, sub_pages)
                return

            parsed_pages = overlay.parse_multi_page_result(ocr_result)
            for idx, page_num in enumerate(sub_pages):
                if idx >= len(parsed_pages):
                    # Страница есть в под-батче, но её нет в ответе OCR — это потеря.
                    logger.warning(
                        f"Page {page_num + 1} missing in OCR result "
                        f"(chunk {chunk_id}, batch {batch_id})"
                    )
                    self._mark_pages_failed([page_num])
                    self._save_empty_pages(chunk_id, [page_num])
                    continue

                self._pages_ok += 1
                text, text_blocks, ocr_page_dim = parsed_pages[idx]
                self.result_writer.save_page_text(chunk_id, page_num, text)
                self.result_writer.save_page_json(chunk_id, page_num, text)

                image_single_page_bytes = read_bytes(per_page_paths[idx])
                try:
                    overlay_pdf_bytes = await self._run_cpu(
                        overlay.create_text_overlay_pdf,
                        image_single_page_bytes,
                        text_blocks,
                        ocr_page_dim,
                        self.font_name,
                    )
                    write_bytes(
                        self.layout.chunk_overlay_pdf_path(chunk_id, page_num), overlay_pdf_bytes
                    )
                finally:
                    del image_single_page_bytes
                    gc.collect()

                logger.info(
                    f"Processed page {page_num + 1} from chunk {chunk_id}, batch {batch_id}"
                )
                del overlay_pdf_bytes
                del text_blocks
                gc.collect()

            del parsed_pages
            del ocr_result
            gc.collect()
        except OCRAuthError:
            raise
        except Exception as exc:
            logger.exception(
                f"Chunk {chunk_id} batch {batch_id} sub-batch failed (pages "
                f"{sub_pages[0] + 1}-{sub_pages[-1] + 1}): {type(exc).__name__}: {exc}"
            )
            self._mark_pages_failed(sub_pages)
            self._save_empty_pages(chunk_id, sub_pages)

    async def process_batch(
        self,
        session: aiohttp.ClientSession,
        chunk_id: int,
        batch_path: Path,
        page_numbers: list[int],
        batch_id: int,
    ) -> None:
        rendered_paths: list[Path] = []
        try:
            sub_batches = await self._run_cpu(
                render.render_batch,
                self.layout,
                batch_path,
                chunk_id,
                batch_id,
                page_numbers,
                self.dpi,
                self.jpeg_quality,
                self.max_batch_bytes,
            )
            for image_pdf_path, sub_pages, per_page_paths in sub_batches:
                rendered_paths.append(image_pdf_path)
                rendered_paths.extend(per_page_paths)
                await self._process_subbatch(
                    session, chunk_id, batch_id, image_pdf_path, sub_pages, per_page_paths
                )
        except OCRAuthError:
            # Авторизация сломана — прерываем весь файл, а не глотаем как пустые страницы.
            raise
        except Exception as exc:
            logger.exception(
                f"Chunk {chunk_id} batch {batch_id} failed (pages "
                f"{page_numbers[0] + 1}-{page_numbers[-1] + 1}): {type(exc).__name__}: {exc}"
            )
            self._mark_pages_failed(page_numbers)
            self._save_empty_pages(chunk_id, page_numbers)
        finally:
            remove_file(batch_path)
            for path in rendered_paths:
                remove_file(path)
            gc.collect()

    async def process_chunk(
        self,
        session: aiohttp.ClientSession,
        chunk_id: int,
        chunk_pdf_path: Path,
        page_numbers: list[int],
        max_concurrent: int,
        batch_size: int,
    ) -> None:
        batches = pdf_split.split_chunk_to_batches(
            self.layout, chunk_id, chunk_pdf_path, page_numbers, batch_size=batch_size
        )

        effective_parallel_batches = 1 if self.strict_memory_mode else max(1, max_concurrent)

        if effective_parallel_batches != 1:
            semaphore = asyncio.Semaphore(effective_parallel_batches)

            async def _run(batch_path: Path, pages: list[int], batch_id: int) -> None:
                async with semaphore:
                    await self.process_batch(session, chunk_id, batch_path, pages, batch_id)

            tasks = [
                asyncio.create_task(_run(batch_path, pages, batch_id))
                for batch_path, pages, batch_id in batches
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Ошибку авторизации пробрасываем — нет смысла продолжать весь файл.
            for r in results:
                if isinstance(r, OCRAuthError):
                    raise r
        else:
            for batch_path, pages, batch_id in batches:
                await self.process_batch(session, chunk_id, batch_path, pages, batch_id)

        self.result_writer.build_chunk_outputs(chunk_id, page_numbers)
        self.result_writer.append_chunk_txt_to_final(chunk_id, is_first_chunk=(chunk_id == 0))
        self.result_writer.append_chunk_json_to_final(chunk_id)

        remove_file(chunk_pdf_path)
        gc.collect()

    async def process_pdf(
        self,
        input_pdf_path: Path,
        max_concurrent: int = 1,
        batch_size: int = 1,
        chunk_size: int = 50,
        cleanup_tmp_files: bool = False,
    ) -> None:
        self.result_writer.init_final_outputs()
        self._pages_ok = 0
        self._pages_failed = 0
        self._failed_pages = []

        started = time.monotonic()
        chunks = pdf_split.split_pdf_to_chunks(self.layout, input_pdf_path, chunk_size=chunk_size)
        total_pages = sum(len(page_numbers) for _, _, page_numbers in chunks)
        logger.info(
            f"Start OCR for {self.base_name}: {total_pages} pages, {len(chunks)} chunks "
            f"(chunk_size={chunk_size}, batch_size={batch_size}, max_concurrent={max_concurrent})"
        )
        timeout = ClientTimeout(total=600, connect=120, sock_connect=120, sock_read=300)
        connector = TCPConnector(limit=max(1, max_concurrent), ttl_dns_cache=300)
        processed_chunk_ids: list[int] = []

        try:
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                for chunk_id, chunk_pdf_path, page_numbers in chunks:
                    logger.info(
                        f"Processing chunk {chunk_id}/{len(chunks) - 1} "
                        f"with {len(page_numbers)} pages "
                        f"(pages {page_numbers[0] + 1}-{page_numbers[-1] + 1})"
                    )
                    chunk_started = time.monotonic()
                    await self.process_chunk(
                        session,
                        chunk_id,
                        chunk_pdf_path,
                        page_numbers,
                        max_concurrent=max_concurrent,
                        batch_size=batch_size,
                    )
                    processed_chunk_ids.append(chunk_id)
                    logger.info(
                        f"Chunk {chunk_id} done in {time.monotonic() - chunk_started:.1f}s"
                    )
                    gc.collect()

            self.result_writer.finalize_final_json()

            logger.info("Building final PDF from chunk files...")
            self._build_final_pdf(processed_chunk_ids)

            logger.info(
                f"OCR finished for {self.base_name}: {total_pages} pages in "
                f"{time.monotonic() - started:.1f}s"
            )
            logger.info(
                f"Pages summary: total={total_pages} ok={self._pages_ok} "
                f"failed={self._pages_failed}"
            )
            logger.info(f"Saved result (txt): {self.layout.txt_result_path()}")
            logger.info(f"Saved result (json): {self.layout.json_result_path()}")
            logger.info(f"Saved result (pdf): {self.layout.pdf_result_path()}")

            # Строгая политика: непустая потеря страниц сверх порога => файл failed,
            # чтобы в индекс не попал неполный документ (его можно перезапустить).
            if self._pages_failed > self.max_failed_pages:
                raise RuntimeError(
                    f"OCR incomplete: {self._pages_failed}/{total_pages} pages failed "
                    f"(allowed {self.max_failed_pages}); failed pages "
                    f"e.g. {self._failed_pages[:30]}"
                )
        except Exception:
            logger.exception(
                f"OCR aborted for {self.base_name} after "
                f"{time.monotonic() - started:.1f}s; processed chunks: {processed_chunk_ids}"
            )
            try:
                self.result_writer.finalize_final_json()
            except Exception:
                pass
            raise
        finally:
            self._cpu.shutdown(wait=True)
            if cleanup_tmp_files:
                self.cleanup_tmp_files()

    def _build_final_pdf(self, chunk_ids: list[int]) -> None:
        output_path = self.layout.pdf_result_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        chunk_pdf_paths = [
            self.layout.chunk_pdf_result_path(chunk_id)
            for chunk_id in chunk_ids
            if self.layout.chunk_pdf_result_path(chunk_id).exists()
        ]

        if not chunk_pdf_paths:
            raise FileNotFoundError("No chunk PDF files were created")

        # Дерево слияний — ограничить память внешних утилит.
        pdf_merge.merge_pdf_files_tree(
            input_paths=chunk_pdf_paths,
            output_path=output_path,
            tmp_dir=self.layout.tmp_dir,
            fan_in=4,
        )
