"""
LoRA fine-tuning for TexTeller.

Ports the Lora_Parametrization-based approach used in the stroke-based
handwriting recognizer to TexTeller's VisionEncoderDecoderModel so per-user
handwriting adapters can be trained in-process from a web-app worker.

The API intentionally mirrors the existing handwriting implementation:
    - `apply_lora` / `remove_lora` / `get_lora_parameters`
    - `extract_lora_state_dict` / `load_lora_state_dict`
    - `lora_to_json` / `lora_from_json`
    - `train_lora_adapter(model, tokenizer, samples, ...)` (async)
"""

import asyncio
import concurrent.futures
import queue
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from texteller.constants import MAX_TOKEN_SIZE
from texteller.utils import transform


# ---------------------------------------------------------------------------
# LoRA primitives
# ---------------------------------------------------------------------------

# TexTeller's decoder.lm_head is hidden_dim x vocab_size (15k); adapting it
# doubles adapter size for little benefit on per-user fine-tuning. Callers
# wanting exact parity with the "every linear" default used in the handwriting
# model can pass exclude_patterns=() to override.
DEFAULT_EXCLUDE_PATTERNS = ("lm_head",)


@dataclass
class Lora_Config:
    rank: int = 8
    alpha: float = 16.0


class Lora_Parametrization(nn.Module):
    def __init__(self, out_features, in_features, rank, alpha):
        super().__init__()
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a)

    def forward(self, weight):
        return weight + self.scale * (self.lora_b @ self.lora_a)


def _get_lora_targets(model, exclude_patterns=DEFAULT_EXCLUDE_PATTERNS):
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(pat in name for pat in exclude_patterns):
            continue
        targets.append((name + ".weight", module, "weight"))
    return targets


def apply_lora(model, config, exclude_patterns=DEFAULT_EXCLUDE_PATTERNS):
    for base_param in model.parameters():
        base_param.requires_grad = False

    for target_name, target_module, param_name in _get_lora_targets(model, exclude_patterns):
        original_weight = getattr(target_module, param_name)
        out_features, in_features = original_weight.shape
        parametrization = Lora_Parametrization(
            out_features, in_features, config.rank, config.alpha
        ).to(original_weight.device)
        parametrize.register_parametrization(
            target_module,
            param_name,
            parametrization,
        )


def remove_lora(model, exclude_patterns=DEFAULT_EXCLUDE_PATTERNS):
    for target_name, target_module, param_name in _get_lora_targets(model, exclude_patterns):
        if parametrize.is_parametrized(target_module, param_name):
            parametrize.remove_parametrizations(
                target_module, param_name, leave_parametrized=False
            )


def get_lora_parameters(model):
    params = []
    for module in model.modules():
        if isinstance(module, Lora_Parametrization):
            params.extend([module.lora_a, module.lora_b])
    return params


def extract_lora_state_dict(model, exclude_patterns=DEFAULT_EXCLUDE_PATTERNS):
    state_dict = {}
    for target_name, target_module, param_name in _get_lora_targets(model, exclude_patterns):
        if not parametrize.is_parametrized(target_module, param_name):
            continue
        parametrization_list = target_module.parametrizations[param_name]
        for p in parametrization_list:
            if isinstance(p, Lora_Parametrization):
                state_dict[target_name + ".lora_a"] = p.lora_a.detach().clone()
                state_dict[target_name + ".lora_b"] = p.lora_b.detach().clone()
    return state_dict


def load_lora_state_dict(model, state_dict, config, exclude_patterns=DEFAULT_EXCLUDE_PATTERNS):
    remove_lora(model, exclude_patterns)
    apply_lora(model, config, exclude_patterns)

    target_map = {}
    for target_name, target_module, param_name in _get_lora_targets(model, exclude_patterns):
        if parametrize.is_parametrized(target_module, param_name):
            parametrization_list = target_module.parametrizations[param_name]
            for p in parametrization_list:
                if isinstance(p, Lora_Parametrization):
                    target_map[target_name] = p

    for target_name, parametrization in target_map.items():
        lora_a_key = target_name + ".lora_a"
        lora_b_key = target_name + ".lora_b"
        if lora_a_key in state_dict:
            parametrization.lora_a.data.copy_(state_dict[lora_a_key])
        if lora_b_key in state_dict:
            parametrization.lora_b.data.copy_(state_dict[lora_b_key])


def lora_to_json(state_dict, config, num_training_samples=0, base_model_hash=""):
    weights = {key: tensor.cpu().tolist() for key, tensor in state_dict.items()}
    return {
        "lora_version": 1,
        "rank": config.rank,
        "alpha": config.alpha,
        "base_model_hash": base_model_hash,
        "num_training_samples": num_training_samples,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights": weights,
    }


def lora_from_json(adapter_json):
    config = Lora_Config(rank=adapter_json["rank"], alpha=adapter_json["alpha"])
    state_dict = {
        key: torch.tensor(values, dtype=torch.float32)
        for key, values in adapter_json["weights"].items()
    }
    return state_dict, config


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

MINIMUM_SAMPLES = 20
DEFAULT_EPOCHS = 40
DEFAULT_LEARNING_RATE = 1e-3
INCREMENTAL_EPOCHS = 20
INCREMENTAL_LEARNING_RATE = 5e-4
L2_ANCHOR_LAMBDA = 0.01


def _prepare_sample(image, label, tokenizer, device):
    image_tensors = transform([image])
    pixel_values = image_tensors[0].unsqueeze(0).to(device)

    token_ids = tokenizer(
        label,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKEN_SIZE - 1,
    ).input_ids.to(device)

    labels = token_ids.clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    return pixel_values, labels


def _run_training(
    model,
    tokenizer,
    samples,
    config,
    epochs,
    learning_rate,
    progress_queue,
    previous_adapter=None,
    exclude_patterns=DEFAULT_EXCLUDE_PATTERNS,
):
    device = next(model.parameters()).device

    prepared_samples = []
    for sample in samples:
        try:
            image, label = sample
            prepared = _prepare_sample(image, label, tokenizer, device)
            prepared_samples.append(prepared)
        except (ValueError, KeyError, TypeError):
            continue

    minimum = 1 if previous_adapter else MINIMUM_SAMPLES
    if len(prepared_samples) < minimum:
        progress_queue.put_nowait({
            "type": "error",
            "message": f"Only {len(prepared_samples)} valid samples (need {minimum})",
        })
        return None

    if previous_adapter:
        previous_state_dict, previous_config = lora_from_json(previous_adapter)
        load_lora_state_dict(model, previous_state_dict, previous_config, exclude_patterns)
    else:
        remove_lora(model, exclude_patterns)
        apply_lora(model, config, exclude_patterns)

    lora_params = get_lora_parameters(model)
    anchor_weights = [p.detach().clone() for p in lora_params] if previous_adapter else None
    optimizer = torch.optim.AdamW(lora_params, lr=learning_rate, weight_decay=0.01)

    try:
        for epoch in range(epochs):
            epoch_loss = 0.0
            for pixel_values, labels in prepared_samples:
                optimizer.zero_grad()
                with torch.enable_grad():
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    if anchor_weights:
                        l2_penalty = sum(
                            ((p - a) ** 2).sum()
                            for p, a in zip(lora_params, anchor_weights)
                        )
                        loss = loss + L2_ANCHOR_LAMBDA * l2_penalty
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            average_loss = epoch_loss / len(prepared_samples)
            progress_queue.put_nowait({
                "type": "progress",
                "epoch": epoch + 1,
                "total_epochs": epochs,
                "loss": round(average_loss, 4),
            })

        lora_state_dict = extract_lora_state_dict(model, exclude_patterns)
        adapter_json = lora_to_json(
            lora_state_dict,
            config,
            num_training_samples=len(prepared_samples),
        )
        return adapter_json

    finally:
        model.eval()


async def train_lora_adapter(
    model,
    tokenizer,
    samples,
    config=None,
    epochs=None,
    learning_rate=None,
    progress_callback=None,
    previous_adapter=None,
    exclude_patterns=DEFAULT_EXCLUDE_PATTERNS,
):
    """
    Train a LoRA adapter on a small handwriting dataset.

    Args:
        model: A TexTeller model (VisionEncoderDecoderModel) already loaded
            via `load_model()`. Modified in place during training.
        tokenizer: The RobertaTokenizerFast from `load_tokenizer()`.
        samples: List of (PIL.Image, latex_str) tuples.
        config: Lora_Config. Defaults to Lora_Config() (rank=8, alpha=16).
        epochs: Override default epoch count. Defaults to
            INCREMENTAL_EPOCHS when `previous_adapter` is given, else
            DEFAULT_EPOCHS.
        learning_rate: Override default learning rate.
        progress_callback: Optional async callable
            `(epoch, total_epochs, loss) -> None` invoked after each epoch.
        previous_adapter: Optional adapter JSON. When given, training
            continues from those weights with an L2 anchor pulling
            parameters back toward the snapshot.
        exclude_patterns: Iterable of substrings; Linear layers whose
            dotted name contains any substring are skipped. Defaults to
            ('lm_head',).

    Returns:
        Adapter JSON dict (see `lora_to_json`) or None if insufficient
        valid samples were provided.
    """
    if config is None:
        config = Lora_Config()
    if epochs is None:
        epochs = INCREMENTAL_EPOCHS if previous_adapter else DEFAULT_EPOCHS
    if learning_rate is None:
        learning_rate = INCREMENTAL_LEARNING_RATE if previous_adapter else DEFAULT_LEARNING_RATE

    sync_queue = queue.Queue()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    future = executor.submit(
        _run_training,
        model,
        tokenizer,
        samples,
        config,
        epochs,
        learning_rate,
        sync_queue,
        previous_adapter,
        exclude_patterns,
    )

    while not future.done():
        await asyncio.sleep(0.5)
        while not sync_queue.empty():
            try:
                message = sync_queue.get_nowait()
                if progress_callback and message.get("type") == "progress":
                    await progress_callback(
                        message["epoch"], message["total_epochs"], message["loss"],
                    )
                elif message.get("type") == "error":
                    raise RuntimeError(message["message"])
            except queue.Empty:
                break

    while not sync_queue.empty():
        try:
            message = sync_queue.get_nowait()
            if progress_callback and message.get("type") == "progress":
                await progress_callback(
                    message["epoch"], message["total_epochs"], message["loss"],
                )
        except queue.Empty:
            break

    result = future.result()
    executor.shutdown(wait=False)
    return result
