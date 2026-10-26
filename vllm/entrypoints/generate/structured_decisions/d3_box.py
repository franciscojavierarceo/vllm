# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D3 localization through yes/no reads of outlined image regions."""

import asyncio
import math
from collections.abc import Mapping
from typing import Any

from PIL import Image, ImageDraw

from vllm.entrypoints.openai.decisions.question_types import build_question
from vllm.entrypoints.openai.decisions.strategies import D3Strategy, cap_pixels
from vllm.exceptions import GenerationError
from vllm.lora.request import LoRARequest
from vllm.multimodal.utils import encode_image_url

from .box import BoxQuestion, BoxRead

GRID_SIZE = 4
GRID_ROUNDS = 4
GRID_READS = GRID_SIZE**2
GRID_MAX_PIXELS = 512**2
MAX_BOX_QUESTIONS = 4


def grid_box(scores: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Enclose cells scoring at least one half, or retain the whole crop."""
    cells = [divmod(i, GRID_SIZE) for i, score in enumerate(scores) if score >= 0.5]
    if not cells:
        return 0, 0, width, height
    rows, cols = zip(*cells)
    return (
        min(cols) * width // GRID_SIZE,
        min(rows) * height // GRID_SIZE,
        ((max(cols) + 1) * width + GRID_SIZE - 1) // GRID_SIZE,
        ((max(rows) + 1) * height + GRID_SIZE - 1) // GRID_SIZE,
    )


def grid_image_urls(image: Image.Image, region: tuple[int, int, int, int]) -> list[str]:
    view = cap_pixels(image.crop(region), GRID_MAX_PIXELS).convert("RGB")
    # Very small images still need sixteen distinct, nonempty cells.
    if min(view.size) < GRID_SIZE:
        view = view.resize((max(GRID_SIZE, view.width), max(GRID_SIZE, view.height)))
    urls = []
    for cell in range(GRID_READS):
        row, col = divmod(cell, GRID_SIZE)
        outlined = view.copy()
        ImageDraw.Draw(outlined).rectangle(
            (
                col * view.width // GRID_SIZE,
                row * view.height // GRID_SIZE,
                (col + 1) * view.width // GRID_SIZE - 1,
                (row + 1) * view.height // GRID_SIZE - 1,
            ),
            outline="red",
            width=max(1, round(min(view.size) / 256)),
        )
        urls.append(encode_image_url(outlined))
    return urls


async def read_d3_boxes(
    strategy: D3Strategy,
    questions: list[BoxQuestion],
    image: Image.Image,
    state: str,
    *,
    request_id: str,
    chat_template_kwargs: dict[str, Any] | None,
    lora_request: LoRARequest | None,
    priority: int,
    cache_salt: str | None,
    trace_headers: Mapping[str, str] | None,
) -> list[BoxRead]:
    """One presence read and four rounds of sixteen cell reads per target.

    The confidence is an uncalibrated presence/cell score, not the hosted
    service's fitted probability of IoU >= 0.5.
    """
    width, height = image.size

    async def read_box(qi: int, question: BoxQuestion) -> BoxRead:
        input_tokens = 0

        async def ask(instructions: str, url: str, suffix: str) -> float:
            nonlocal input_tokens
            q = build_question("cell", "noul", instructions, None, 2, labeled=False)
            (read,) = await strategy.read(
                [q],
                None,
                state,
                request_id=f"{request_id}-box-{qi}-{suffix}",
                chat_template_kwargs=chat_template_kwargs,
                lora_request=lora_request,
                priority=priority,
                cache_salt=cache_salt,
                trace_headers=trace_headers,
                media={"image": [url]},
            )
            input_tokens += read.input_tokens
            probability = read.probs[0]
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise GenerationError(f"Invalid Box score for question {question.id!r}")
            return probability

        presence_url = await asyncio.to_thread(
            lambda: encode_image_url(cap_pixels(image, GRID_MAX_PIXELS))
        )
        presence = await ask(
            f"Is {question.instructions} visible anywhere in the image?",
            presence_url,
            "presence",
        )
        region = (0, 0, width, height)
        confidence = presence
        for step in range(GRID_ROUNDS):
            crop = region
            if step:
                x0, y0, x1, y1 = region
                dx, dy = max(1, (x1 - x0) // 8), max(1, (y1 - y0) // 8)
                crop = (
                    max(0, x0 - dx),
                    max(0, y0 - dy),
                    min(width, x1 + dx),
                    min(height, y1 + dy),
                )
            left, top, right, bottom = crop
            urls = await asyncio.to_thread(grid_image_urls, image, crop)
            tasks = [
                asyncio.create_task(
                    ask(
                        f"Is any part of {question.instructions} inside the red "
                        "rectangle? Ignore the rectangle itself.",
                        url,
                        f"round-{step}-cell-{cell}",
                    )
                )
                for cell, url in enumerate(urls)
            ]
            try:
                scores = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            selected = [score for score in scores if score >= 0.5]
            if selected:
                x0, y0, x1, y1 = grid_box(scores, right - left, bottom - top)
                region = (left + x0, top + y0, left + x1, top + y1)
            confidence = presence * (
                sum(selected) / len(selected) if selected else max(scores)
            )
        left, top, right, bottom = region
        return BoxRead(
            answer={
                "type": "box",
                "box": {
                    "left": left / width,
                    "top": top / height,
                    "right": right / width,
                    "bottom": bottom / height,
                },
                "confidence": confidence,
            },
            input_tokens=input_tokens,
            output_tokens=0,
        )

    return [await read_box(qi, question) for qi, question in enumerate(questions)]
