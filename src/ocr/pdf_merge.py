"""Слияние постраничных PDF в итоговый: внешняя утилита (pdfunite/qpdf/gs) с
фолбэком на pypdf; дерево слияний для ограничения памяти внешних утилит."""
import gc
import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from logging_config import get_logger

logger = get_logger("ocr")


def _run_merge_cmd(cmd: list[str], tool: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # -q/-dQUIET глушат вывод gs, поэтому логируем то, что есть.
        logger.error(
            f"{tool} merge failed (exit {result.returncode}): "
            f"stderr={result.stderr.strip()[:300]} stdout={result.stdout.strip()[:200]}"
        )
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)


def merge_pdfs_pypdf(input_paths: list[Path], output_path: Path) -> None:
    """Фолбэк-слияние через pypdf (постранично, без внешних утилит)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for path in input_paths:
        try:
            with path.open("rb") as src:
                reader = PdfReader(src)
                for page in reader.pages:
                    writer.add_page(page)
        except Exception as exc:
            logger.warning(f"pypdf merge: skipping {path}: {exc}")
    with output_path.open("wb") as dst:
        writer.write(dst)


def run_pdf_merge_tool(input_paths: list[Path], output_path: Path) -> str:
    if not input_paths:
        raise ValueError("input_paths is empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdfunite_path = shutil.which("pdfunite")
    if pdfunite_path:
        _run_merge_cmd(
            [pdfunite_path, *[str(p) for p in input_paths], str(output_path)], "pdfunite"
        )
        return "pdfunite"

    qpdf_path = shutil.which("qpdf")
    if qpdf_path:
        _run_merge_cmd(
            [
                qpdf_path,
                "--empty",
                "--pages",
                *[str(p) for p in input_paths],
                "--",
                str(output_path),
            ],
            "qpdf",
        )
        return "qpdf"

    gs_path = shutil.which("gs")
    if gs_path:
        _run_merge_cmd(
            [
                gs_path,
                "-dBATCH",
                "-dNOPAUSE",
                "-q",
                "-sDEVICE=pdfwrite",
                f"-sOutputFile={output_path}",
                *[str(p) for p in input_paths],
            ],
            "ghostscript",
        )
        return "ghostscript"

    raise RuntimeError("No PDF merge tool found: pdfunite, qpdf, gs")


def merge_pdfs(input_paths: list[Path], output_path: Path) -> str:
    """Слить PDF внешней утилитой; при её отсутствии/ошибке — фолбэк на pypdf."""
    try:
        return run_pdf_merge_tool(input_paths, output_path)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        logger.warning(f"External PDF merge failed ({exc}); falling back to pypdf")
        merge_pdfs_pypdf(input_paths, output_path)
        return "pypdf"


def merge_pdf_files_tree(
    input_paths: list[Path],
    output_path: Path,
    tmp_dir: Path,
    fan_in: int = 4,
) -> None:
    if not input_paths:
        raise FileNotFoundError("No input PDF files to merge")
    if fan_in < 2:
        raise ValueError("fan_in must be >= 2")

    current_paths = input_paths[:]
    merge_tmp_dir = tmp_dir / "merge-rounds"
    merge_tmp_dir.mkdir(parents=True, exist_ok=True)
    round_index = 0

    while len(current_paths) > 1:
        next_paths: list[Path] = []
        logger.info(
            f"Final PDF merge round {round_index}: "
            f"{len(current_paths)} files, fan_in={fan_in}"
        )

        for group_index in range(0, len(current_paths), fan_in):
            group = current_paths[group_index: group_index + fan_in]
            if len(group) == 1:
                next_paths.append(group[0])
                continue

            intermediate_path = (
                merge_tmp_dir
                / f"round_{round_index:04d}_group_{group_index // fan_in:04d}.pdf"
            )
            tool_name = merge_pdfs(group, intermediate_path)
            logger.info(
                f"Merged round={round_index} group={group_index // fan_in} "
                f"with {tool_name}: {len(group)} -> {intermediate_path.name}"
            )
            next_paths.append(intermediate_path)

        for old_path in current_paths:
            try:
                if old_path not in input_paths and old_path.exists():
                    old_path.unlink()
            except Exception as exc:
                logger.warning(f"Failed to delete intermediate merge file {old_path}: {exc}")

        current_paths = next_paths
        round_index += 1
        gc.collect()

    final_candidate = current_paths[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    if final_candidate == output_path:
        logger.info(f"Saved final PDF: {output_path}")
        return

    final_candidate.replace(output_path)
    logger.info(f"Saved final PDF: {output_path}")
