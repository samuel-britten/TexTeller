from __future__ import annotations

import base64
import io
from typing import Any, Awaitable, Callable, Optional

import torch
from PIL import Image

from texteller import (
    Lora_Config,
    apply_lora,
    img2latex,
    load_lora_state_dict,
    load_model,
    load_tokenizer,
    lora_from_json,
    remove_lora,
    train_lora_adapter,
)


ProgressCallback = Callable[[int, int, float], Awaitable[None]]


class TexTellerRecognizer:
    """
    Stateful TexTeller wrapper. Instantiate once at server startup and
    reuse across all requests — model loading is slow, inference is cheap.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.lora_active = False
        self.training_in_progress = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_dir: Optional[str] = None,
        tokenizer_dir: Optional[str] = None,
        device: str | torch.device | None = None,
    ) -> "TexTellerRecognizer":
        """
        Load a TexTellerRecognizer.

        Args:
            model_dir:     Override the default HF Hub model. None = default.
            tokenizer_dir: Override the default tokenizer.        None = default.
            device:        "cpu", "cuda", torch.device, or None for auto.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        model = load_model(model_dir=model_dir).to(device)
        model.eval()
        tokenizer = load_tokenizer(tokenizer_dir=tokenizer_dir)
        return cls(model=model, tokenizer=tokenizer, device=device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recognize(
        self,
        image: Image.Image,
        num_beams: int = 1,
    ) -> str:
        """
        Convert a PIL image into a LaTeX string.

        Works identically whether a LoRA adapter is currently active or not.
        """
        import numpy as np

        arr = np.array(image.convert("RGB"))
        results = img2latex(
            self.model,
            self.tokenizer,
            [arr],
            device=self.device,
            num_beams=num_beams,
        )
        return results[0]

    # ------------------------------------------------------------------
    # LoRA adapter management
    # ------------------------------------------------------------------

    def apply_adapter(self, adapter_json: dict) -> None:
        """Load a serialized adapter onto the live base model."""
        state_dict, config = lora_from_json(adapter_json)
        load_lora_state_dict(self.model, state_dict, config)
        self.model.eval()
        self.lora_active = True

    def remove_adapter(self) -> None:
        """Strip any active adapter; model returns to vanilla base behavior."""
        remove_lora(self.model)
        self.model.eval()
        self.lora_active = False

    async def train_adapter(
        self,
        samples: list[tuple[Image.Image, str]],
        progress_callback: Optional[ProgressCallback] = None,
        previous_adapter: Optional[dict] = None,
        config: Optional[Lora_Config] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Train a LoRA adapter on a list of (PIL.Image, latex_str) samples.

        Returns the adapter JSON (same schema your web app already persists
        to Google Drive), or None if too few valid samples were provided.
        Raises RuntimeError if training is already running or if the
        background worker reports an error.

        `progress_callback` is the same shape as your handwriting version:
            async def cb(epoch: int, total_epochs: int, loss: float): ...

        Pass `previous_adapter` for incremental training with the L2 anchor
        — the loaded weights are pulled back toward the snapshot so the
        adapter doesn't drift on small batches of new samples.
        """
        if self.training_in_progress:
            raise RuntimeError("Training already in progress")

        self.training_in_progress = True
        try:
            adapter_json = await train_lora_adapter(
                self.model,
                self.tokenizer,
                samples,
                config=config,
                epochs=epochs,
                learning_rate=learning_rate,
                progress_callback=progress_callback,
                previous_adapter=previous_adapter,
            )
            if adapter_json is not None:
                self.lora_active = True
                self.model.eval()
            return adapter_json
        finally:
            self.training_in_progress = False


# ---------------------------------------------------------------------------
# Reference WebSocket handler — mirrors the contract your Lora_Manager.js
# frontend already uses. Adapt to whichever framework you're on (FastAPI,
# aiohttp, starlette, ...). Nothing here is required by TexTellerRecognizer.
# ---------------------------------------------------------------------------

def _decode_sample(sample: dict) -> tuple[Image.Image, str]:
    """
    Reference decoder for sample dicts coming over the wire. Adjust to match
    whatever your frontend actually sends. Expected shape:
        {"image": "<base64 png bytes>", "label": "<latex string>"}
    """
    raw = base64.b64decode(sample["image"])
    image = Image.open(io.BytesIO(raw))
    return image, sample["label"]


async def handle_ws_message(
    recognizer: TexTellerRecognizer,
    message: dict[str, Any],
    send: Callable[[dict], Awaitable[None]],
) -> None:
    """Map incoming WebSocket messages to recognizer operations."""
    mtype = message.get("type")

    if mtype == "lora_load_adapter":
        try:
            recognizer.apply_adapter(message["adapter"])
            await send({"type": "lora_adapter_loaded", "success": True})
        except Exception as e:
            await send({"type": "lora_adapter_loaded", "success": False, "error": str(e)})

    elif mtype == "lora_remove_adapter":
        recognizer.remove_adapter()
        await send({"type": "lora_adapter_removed"})

    elif mtype == "lora_train":
        samples = [_decode_sample(s) for s in message["samples"]]

        async def on_progress(epoch: int, total_epochs: int, loss: float) -> None:
            await send({
                "type": "lora_training_progress",
                "epoch": epoch,
                "total_epochs": total_epochs,
            })

        try:
            adapter_json = await recognizer.train_adapter(
                samples,
                progress_callback=on_progress,
                previous_adapter=message.get("previous_adapter"),
            )
            if adapter_json is None:
                await send({
                    "type": "lora_training_error",
                    "error": "Insufficient valid samples",
                })
            else:
                await send({
                    "type": "lora_training_complete",
                    "adapter": adapter_json,
                })
        except Exception as e:
            await send({"type": "lora_training_error", "error": str(e)})
