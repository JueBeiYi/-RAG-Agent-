from __future__ import annotations

import pytest

from src.retrieval.token_windows import single_windows, pair_windows


class Tokenizer:
    """Roundtrip deliberately unavailable, original unknown ID must survive."""
    def encode(self, text, **_):
        return [0 if item == "unknown" else int(item) for item in text.split()]
    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2
    def prepare_for_model(self, ids, pair_ids=None, **_):
        values = [101, *ids, 102]
        if pair_ids is not None:
            values.extend([*pair_ids, 102])
        return {"input_ids": values, "attention_mask": [1] * len(values)}
    def decode(self, *_):
        raise AssertionError("Must not decode original IDs")


def test_single_windows_cover_tail_and_unk_without_retokenising():
    text = " ".join([str(i) for i in range(15)] + ["unknown", "16", "17"])
    windows = single_windows(Tokenizer(), [text], 8)
    contents = [i for feature in windows.features for i in feature["input_ids"][1:-1]]
    assert contents == list(range(15)) + [0, 16, 17]
    assert windows.token_ranges == [(0, 6), (6, 12), (12, 18)]
    assert windows.weights == [6, 6, 6]
    assert all(len(feature["input_ids"]) <= 8 for feature in windows.features)


def test_pair_windows_repeat_query_but_cover_every_document_id_once():
    windows = pair_windows(Tokenizer(), "90 91", [" ".join(str(i) for i in range(23))], 15)
    assert windows.source_token_counts == [23]
    assert windows.token_ranges == [(0, 10), (10, 20), (20, 23)]
    assert [i for feature in windows.features for i in feature["input_ids"][4:-1]] == list(range(23))
    assert all(feature["input_ids"][:4] == [101, 90, 91, 102] for feature in windows.features)
    assert all(len(feature["input_ids"]) <= 15 for feature in windows.features)


def test_hf5_rust_post_processor_keeps_overflow_ids_and_pair_segments():
    tokenizers = pytest.importorskip("tokenizers")
    vocabulary = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, **{f"t{i}": i + 3 for i in range(30)}}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    class HF5:
        backend_tokenizer = backend
        model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
        def num_special_tokens_to_add(self, pair=False):
            return 3 if pair else 2
        def encode(self, text, **_):
            return backend.encode(text, add_special_tokens=False).ids
    wrapper = HF5()
    document = " ".join([f"t{i}" for i in range(23)] + ["unseen"])
    actual_ids = wrapper.encode(document)
    windows = single_windows(wrapper, [document], 8)
    assert [i for row in windows.features for i in row["input_ids"][1:-1]] == actual_ids
    paired = pair_windows(wrapper, "t25 t26", [document], 15)
    assert [i for row in paired.features for i in row["input_ids"][4:-1]] == actual_ids
    assert all(row["token_type_ids"][:4] == [0, 0, 0, 0] for row in paired.features)
    assert all(set(row["token_type_ids"][4:]) == {1} for row in paired.features)
