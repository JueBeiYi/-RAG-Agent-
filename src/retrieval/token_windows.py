"""Lossless original-token windows shared by embedding and reranking.

WordPiece fragments cannot safely be decoded into strings and re-tokenized at
arbitrary boundaries. These helpers preserve original IDs (including UNK) and
add only the tokenizer's required special tokens before model inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WINDOW_POLICY = "original-token-id-windows-v3"


@dataclass
class TokenWindows:
    features: list[dict]
    owners: list[int]
    weights: list[int]
    token_ranges: list[tuple[int, int]]
    source_token_counts: list[int]
    limit: int

    @property
    def long_input_count(self):
        return sum(sum(owner == i for owner in self.owners) > 1 for i in range(len(self.source_token_counts)))


def _prepared_windows(tokenizer, text, available, query=None):
    """Support HF4 prepare_for_model and HF5 Rust post-processor natively."""
    if hasattr(tokenizer, "prepare_for_model"):
        ids = tokenizer.encode(text.strip(), add_special_tokens=False, truncation=False)
        query_ids = tokenizer.encode(query.strip(), add_special_tokens=False, truncation=False) if query is not None else None
        rows = []
        for start in range(0, max(1, len(ids)), available):
            end = min(start + available, len(ids))
            if query_ids is None:
                prepared = tokenizer.prepare_for_model(ids[start:end], add_special_tokens=True,
                                                       truncation=False, return_attention_mask=True)
            else:
                prepared = tokenizer.prepare_for_model(query_ids, pair_ids=ids[start:end], add_special_tokens=True,
                                                       truncation=False, return_attention_mask=True)
            rows.append((dict(prepared), start, end))
        return rows, len(ids)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise RuntimeError("Tokenizer缺少原token-ID特殊token处理接口，拒绝decode重编码或截断")
    backend.no_truncation()
    backend.no_padding()
    encoding = backend.encode(text.strip(), add_special_tokens=False)
    original = list(encoding.ids)
    query_encoding = backend.encode(query.strip(), add_special_tokens=False) if query is not None else None
    encoding.truncate(available, stride=0)
    windows = [encoding, *encoding.overflowing]
    if [identifier for window in windows for identifier in window.ids] != original:
        raise RuntimeError("Rust tokenizer窗口未精确覆盖全部原token IDs")
    rows, start = [], 0
    for window in windows:
        length = len(window.ids)
        processed = backend.post_process(window, add_special_tokens=True) if query_encoding is None else backend.post_process(query_encoding, window, add_special_tokens=True)
        prepared = {"input_ids": list(processed.ids), "attention_mask": list(processed.attention_mask)}
        if "token_type_ids" in tokenizer.model_input_names:
            prepared["token_type_ids"] = list(processed.type_ids)
        rows.append((prepared, start, start + length))
        start += length
    return rows, len(original)


def single_windows(tokenizer, texts: list[str], limit: int) -> TokenWindows:
    available = limit - tokenizer.num_special_tokens_to_add(pair=False)
    if available < 1:
        raise ValueError("Embedding窗口不足以容纳必要特殊token")
    features, owners, weights, ranges, counts = [], [], [], [], []
    for owner, text in enumerate(texts):
        rows, count = _prepared_windows(tokenizer, text, available)
        counts.append(count)
        for prepared, start, end in rows:
            if len(prepared["input_ids"]) > limit:
                raise ValueError("Tokenizer添加特殊token后超过模型窗口，拒绝截断")
            features.append(dict(prepared))
            owners.append(owner)
            weights.append(max(1, end - start))
            ranges.append((start, end))
    return TokenWindows(features, owners, weights, ranges, counts, limit)


def pair_windows(tokenizer, query: str, documents: list[str], limit: int) -> TokenWindows:
    query_ids = tokenizer.encode(query.strip(), add_special_tokens=False, truncation=False)
    available = limit - len(query_ids) - tokenizer.num_special_tokens_to_add(pair=True)
    if available < 8:
        raise ValueError("问题超过重排模型输入限制；请精简问题或使用hybrid模式")
    features, owners, weights, ranges, counts = [], [], [], [], []
    for owner, text in enumerate(documents):
        rows, count = _prepared_windows(tokenizer, text, available, query=query)
        counts.append(count)
        for prepared, start, end in rows:
            if len(prepared["input_ids"]) > limit:
                raise ValueError("Tokenizer成对特殊token超过模型窗口，拒绝截断")
            features.append(dict(prepared))
            owners.append(owner)
            weights.append(max(1, end - start))
            ranges.append((start, end))
    return TokenWindows(features, owners, weights, ranges, counts, limit)


def sentence_vectors(model, windows: TokenWindows, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    output = None
    totals = np.zeros(len(windows.source_token_counts), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(windows.features), max(1, batch_size)):
            features = model.tokenizer.pad(windows.features[start:start + batch_size], padding=True, return_tensors="pt")
            features = {key: value.to(model.device) for key, value in features.items()}
            encoded = model(features)["sentence_embedding"].float().cpu().numpy()
            encoded /= np.maximum(np.linalg.norm(encoded, axis=1, keepdims=True), 1e-12)
            if output is None:
                output = np.zeros((len(windows.source_token_counts), encoded.shape[1]), dtype=np.float32)
            for position, vector in enumerate(encoded, start):
                owner, weight = windows.owners[position], windows.weights[position]
                output[owner] += vector * weight
                totals[owner] += weight
    if output is None:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    output /= np.maximum(totals[:, None], 1)
    return output / np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)


def classifier_scores(model, tokenizer, windows: TokenWindows, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    device = next(model.parameters()).device
    scores = []
    with torch.inference_mode():
        for start in range(0, len(windows.features), max(1, batch_size)):
            features = tokenizer.pad(windows.features[start:start + batch_size], padding=True, return_tensors="pt")
            features = {key: value.to(device) for key, value in features.items()}
            raw = model(**features).logits.float().cpu().numpy()
            if raw.ndim != 2 or raw.shape[1] != 1:
                raise ValueError("科研重排适配器需要单标量logit分类头")
            scores.extend(raw[:, 0].tolist())
    return np.asarray(scores, dtype=np.float32)
