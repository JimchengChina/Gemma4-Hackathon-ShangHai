from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schemas import EvidenceFrame


AUDIT_LOG_FILE = "audit_log.jsonl"

PHONE_PATTERN = re.compile(r"\b1\d{10}\b")
ID_PATTERN = re.compile(r"\b\d{17}[\dXx]\b")
PLATE_PATTERN = re.compile(r"[\u4e00-\u9fff][A-Z][A-Z0-9]{5,7}|\b[A-Z]{1}[A-Z0-9]{5,7}\b")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def append_audit_log(payload: dict[str, Any], log_file: str = AUDIT_LOG_FILE) -> None:
    payload = dict(payload)
    payload["logged_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def redact_text(text: str) -> str:
    """Mask common private identifiers before model inference."""
    text = PHONE_PATTERN.sub("[PHONE]", text)
    text = ID_PATTERN.sub("[IDCARD]", text)
    text = PLATE_PATTERN.sub("[PLATE]", text)
    return text


def save_redacted_image(src_path: str, out_dir: str = "redacted") -> str:
    """Placeholder image redaction hook.

    Replace this with face/license-plate detectors in production. The demo keeps
    the image unchanged so the pipeline remains easy to inspect.
    """
    from PIL import Image

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    image = Image.open(src_path).convert("RGB")
    out_path = str(Path(out_dir) / Path(src_path).name)
    image.save(out_path)
    return out_path


def extract_video_metadata(video_path: str) -> dict[str, Any]:
    """
    Extract basic video metadata.

    Args:
        video_path: Local accident video path.

    Returns:
        FPS, frame count, duration and SHA-256 hash.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()

    duration_sec = round(float(frame_count) / float(fps), 3) if fps else None
    return {
        "fps": round(float(fps), 3),
        "frame_count": int(frame_count),
        "duration_sec": duration_sec,
        "sha256": sha256_file(video_path),
    }


def sample_keyframes(
    video_path: str,
    sample_every_sec: float = 1.0,
    max_frames: int = 24,
    out_dir: str = "frames",
) -> list[EvidenceFrame]:
    """Sample frames at fixed intervals.

    Production systems should add scene-cut detection, collision-near oversampling
    and object-track continuity checks.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    import cv2

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(int(fps * sample_every_sec), 1)

    frames: list[EvidenceFrame] = []
    frame_index = 0
    saved = 0

    while cap.isOpened() and saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_interval == 0:
            timestamp_sec = frame_index / fps
            frame_path = str(Path(out_dir) / f"{uuid.uuid4().hex}_{saved:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(EvidenceFrame(image_path=frame_path, timestamp_sec=timestamp_sec))
            saved += 1
        frame_index += 1

    cap.release()
    return frames


def preprocess_multimodal_case(
    video_path: str,
    text_report: str,
    image_paths: list[str],
) -> dict[str, Any]:
    """
    Redact text/images and sample keyframes from the accident video.

    Args:
        video_path: Local accident video path.
        text_report: Police report, user statement or claim narrative.
        image_paths: Local accident scene image paths.

    Returns:
        Redacted text, redacted image paths, sampled frame records and metadata.
    """
    redacted_images = [
        save_redacted_image(path)
        for path in image_paths
        if os.path.exists(path)
    ]
    frames = sample_keyframes(video_path=video_path, sample_every_sec=1.0, max_frames=12)

    return {
        "redacted_text_report": redact_text(text_report),
        "redacted_images": redacted_images,
        "sampled_frames": [asdict(frame) for frame in frames],
        "video_metadata": extract_video_metadata(video_path),
    }


def retrieve_law_articles(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Retrieve traffic-law articles for liability reasoning.

    Args:
        query: Search query created from visible facts or report text.
        top_k: Maximum returned articles.

    Returns:
        Law article candidates. Replace this mock with hybrid RAG in production.
    """
    _ = query
    mock_laws = [
        {
            "article_id": "road-traffic-law-43",
            "title": "安全车距",
            "text": "同车道行驶，后车应当与前车保持足以采取紧急制动措施的安全距离。",
        },
        {
            "article_id": "road-traffic-law-47",
            "title": "人行横道让行",
            "text": "机动车行经人行横道时，应当减速行驶；遇行人通过人行横道，应当停车让行。",
        },
        {
            "article_id": "road-traffic-law-52",
            "title": "事故现场处置",
            "text": "机动车发生交通事故后应当立即停车，保护现场；造成人身伤亡的应立即抢救伤者。",
        },
    ]
    return mock_laws[:top_k]


def calibrated_ratio_head(candidate_bucket: str, parties: list[str]) -> dict[str, float]:
    """
    Map a liability bucket to a conservative ratio template.

    Args:
        candidate_bucket: 全责, 主责, 同责, 次责 or 无责.
        parties: Ordered party names. For two-party cases, the bucket is from
            the first party's perspective.

    Returns:
        Liability ratios whose values sum approximately to 1.0.
    """
    if not parties:
        return {}

    if len(parties) != 2:
        share = round(1.0 / len(parties), 4)
        return {party: share for party in parties}

    first, second = parties
    mapping = {
        "全责": {first: 1.0, second: 0.0},
        "主责": {first: 0.7, second: 0.3},
        "同责": {first: 0.5, second: 0.5},
        "次责": {first: 0.3, second: 0.7},
        "无责": {first: 0.0, second: 1.0},
    }
    return mapping.get(candidate_bucket, mapping["同责"])
