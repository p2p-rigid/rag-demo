from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

import httpx

from .store import StoredChunk


@dataclass(frozen=True)
class DeepSeekClient:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 60.0

    def messages(self, question: str, chunks: list[StoredChunk]) -> list[dict[str, str]]:
        numbered_context = format_context(chunks)
        return [
            {
                "role": "system",
                "content": (
                    "You are an insurance document assistant. Use only the provided "
                    "Zurich document excerpts. Do not invent policy terms. If the "
                    "excerpts do not contain enough information, say what is missing. "
                    "Do not give legal, financial, or insurance advice. Cite source "
                    "labels from the provided context. Keep the answer concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Context:\n{numbered_context}\n\n"
                    "Return exactly these sections:\n"
                    "Answer:\n"
                    "Relevant conditions / limitations:\n"
                    "Sources:\n\n"
                    "Use short bullets where helpful. Do not include more than "
                    "8 bullets in total."
                ),
            },
        ]

    def payload(self, question: str, chunks: list[StoredChunk], stream: bool = False) -> dict:
        payload = {
            "model": self.model,
            "messages": self.messages(question, chunks),
            "temperature": 0.1,
            "max_tokens": 1200,
            "thinking": {"type": "disabled"},
        }
        if stream:
            payload["stream"] = True
        return payload

    def chat(self, question: str, chunks: list[StoredChunk]) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        response = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=self.payload(question, chunks),
            timeout=self.timeout_seconds,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text[:1000]
            raise RuntimeError(
                f"DeepSeek request failed with HTTP {response.status_code}: {detail}"
            ) from exc
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def stream_chat(self, question: str, chunks: list[StoredChunk]) -> Iterator[str]:
        url = self.base_url.rstrip("/") + "/chat/completions"
        with httpx.stream(
            "POST",
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=self.payload(question, chunks, stream=True),
            timeout=self.timeout_seconds,
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.read().decode("utf-8", errors="replace")[:1000]
                raise RuntimeError(
                    f"DeepSeek stream failed with HTTP {response.status_code}: {detail}"
                ) from exc

            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                try:
                    event = __import__("json").loads(data)
                except Exception:
                    continue
                delta = event.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content


def source_label(index: int, chunk: StoredChunk) -> str:
    return (
        f"[{index}] {chunk.title} | {chunk.product} | "
        f"{chunk.document_type} | chunk {chunk.chunk_index}"
    )


def format_context(chunks: list[StoredChunk], max_chars_per_chunk: int = 1400) -> str:
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        text = chunk.text.strip()
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rstrip() + "\n[truncated]"
        parts.append(f"{source_label(index, chunk)}\n{text}")
    return "\n\n---\n\n".join(parts)
