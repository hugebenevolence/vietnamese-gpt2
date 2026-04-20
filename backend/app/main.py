from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)


class ChatResponse(BaseModel):
    reply: str
    backend: str


def _build_prompt(req: ChatRequest) -> str:
    lines: list[str] = []
    for msg in req.history:
        speaker = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{speaker}: {msg.content.strip()}")
    lines.append(f"User: {req.message.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def get_generator() -> Any:
    model_path = os.getenv("MODEL_PATH", "artifacts/model_stage2")
    try:
        from transformers import pipeline

        return pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path,
            device_map="auto",
        )
    except Exception:
        return None


app = FastAPI(title="Vietnamese GPT-2 Chat API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    generator = get_generator()

    if generator is None:
        return ChatResponse(
            reply=f"(mock) Bạn vừa nói: {req.message}",
            backend="mock",
        )

    prompt = _build_prompt(req)
    pad_token_id = getattr(generator.tokenizer, "eos_token_id", None)

    try:
        output = generator(
            prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=True,
            pad_token_id=pad_token_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    text = output[0]["generated_text"]
    reply = text[len(prompt) :].strip() or text.strip()
    return ChatResponse(reply=reply, backend="transformers")
