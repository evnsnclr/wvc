"""``wvc diff --format agent``: the wvc-diff-agent/0 Markdown view (spec section 2.2)."""

import json
import re

import numpy as np
import pytest
from helpers import write_safetensors
from test_hub import _pair as hub_pair
from test_hub import _publish

from wvc.agent_report import AGENT_FORMAT, MAX_LAYER_ROWS, code, ranges, render_agent, split_name
from wvc.cli import main

VECTORS = __import__("pathlib").Path(__file__).resolve().parents[1] / "spec" / "vectors" / "v0"
SECTIONS = [
    "## Pair",
    "## Headline",
    "## Changed tensors by projection",
    "## Changed tensors by layer",
    "## Largest changes",
    "## Added, removed, reshaped",
    "## What this does and does not show",
]
LINE_BUDGET = 150
PROJECTIONS = (
    "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight", "post_attention_layernorm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "mlp.extra.weight",
)  # fmt: skip


def _agent(capsys, *args):
    status = main(["diff", "--format", "agent", *map(str, args)])
    return status, capsys.readouterr().out


def _section_order(text):
    return [line for line in text.splitlines() if line.startswith("## ")]


def test_the_spec_vectors_render_in_the_stable_layout(capsys):
    status, text = _agent(capsys, "--rank", VECTORS / "a.safetensors", VECTORS / "b.safetensors")
    assert status == 1
    assert text.startswith(f"# wvc diff: agent view ({AGENT_FORMAT})\n")
    assert [line.split(" (")[0] for line in _section_order(text)] == SECTIONS
    assert "- tensors: 4 (1 changed, 1 unchanged, 1 added, 1 removed, 0 reshaped)" in text
    assert "- elements changed: 4 of 32 (12.50%)" in text
    assert "| `w` | F32 | [4, 6] | 4 of 24 (16.67%) | 4 x 1 | 0.418 | 1 | 1 | 1 |" in text
    assert "  - `x` I8 [3]" in text
    assert "  - `h` F16 [2, 2, 2]" in text
    assert "- A diff shows where a model changed and how, not what the change does." in text
    assert "quantization comparison" not in text


def test_the_gguf_vector_names_the_comparison(capsys):
    pair = (VECTORS / "quant-base.safetensors", VECTORS / "quant.gguf")
    status, text = _agent(capsys, "--rank", *pair)
    assert status == 1
    assert "names mapped by the `llama` table" in text
    assert "GGUF v3, architecture `llama`" in text
    assert "| `model.layers.*.self_attn.q_proj.weight` | 0 (1) | BF16->Q4_K |" in text
    assert "This is a quantization comparison" in text


def test_the_view_agrees_with_the_json_record(capsys):
    pair = (VECTORS / "a.safetensors", VECTORS / "b.safetensors")
    assert main(["diff", "--rank", "--json", "--all", *map(str, pair)]) == 1
    record = json.loads(capsys.readouterr().out)
    _, text = _agent(capsys, "--rank", *pair)
    assert text == render_agent(record, {"label": "files of b whose tensors are all unchanged",
                                         "names": [], "total": 1})  # fmt: skip


def _big_pair(tmp_path, rng, layers=70, alternate=False):
    """700 tensors (70 layers of 10): every layer's q_proj gets a rank-1 edit, the rest dense
    noise; with ``alternate``, odd layers leave ``mlp.extra`` unchanged."""
    base, edit = {}, {}
    for layer in range(layers):
        for projection in PROJECTIONS:
            name = f"model.layers.{layer}.{projection}"
            values = rng.standard_normal((4, 8) if "norm" not in projection else 8)
            values = values.astype(np.float32)
            if "q_proj" in projection:
                changed = values + np.outer(rng.standard_normal(4), rng.standard_normal(8))
            elif alternate and layer % 2 and "extra" in projection:
                changed = values
            else:
                changed = values + rng.standard_normal(values.shape) * 0.01
            base[name] = ("F32", values)
            edit[name] = ("F32", changed.astype(np.float32))
    a = write_safetensors(tmp_path / "a.safetensors", base)
    b = write_safetensors(tmp_path / "b.safetensors", edit)
    return a, b


def test_a_700_tensor_model_fits_the_line_budget(tmp_path, rng, capsys):
    a, b = _big_pair(tmp_path, rng)
    status, text = _agent(capsys, "--rank", a, b)
    assert status == 1
    lines = text.splitlines()
    assert len(lines) <= LINE_BUDGET, len(lines)
    assert "- tensors: 700 (700 changed, 0 unchanged, 0 added, 0 removed, 0 reshaped)" in text
    assert "- layers with a changed tensor: 70 of 70" in text
    assert "| `model.layers.0-69` | 700 | all 10 | 19,040 of 19,040 (100.00%) |" in text
    assert "| and 688 more changed tensors |" in text
    q = next(line for line in lines if line.startswith("| `model.layers.*.self_attn.q_proj"))
    assert q.split(" | ")[1] == "0-69 (70)"
    assert q.split(" | ")[5] == "1"  # every q_proj edit is rank 1


def test_layer_runs_are_truncated_with_a_count(tmp_path, rng, capsys):
    a, b = _big_pair(tmp_path, rng, alternate=True)
    _, text = _agent(capsys, a, b)
    assert len(text.splitlines()) <= LINE_BUDGET
    assert "| `model.layers.1` | 9 | all 10 but `mlp.extra.weight` |" in text
    rest = 70 - MAX_LAYER_ROWS
    assert f"| and {rest} more layer runs ({rest // 2 * 10 + rest // 2 * 9} tensors) |" in text
    assert "- tensors: 700 (665 changed, 35 unchanged" in text


def test_an_unchanged_pair_says_so(tmp_path, rng, capsys):
    values = rng.standard_normal(4).astype(np.float32)
    a = write_safetensors(tmp_path / "a.safetensors", {"w": ("F32", values)})
    status, text = _agent(capsys, a, a)
    assert status == 0
    assert "## Changed tensors\n- none" in text
    assert "- files of b whose tensors are all unchanged: 1 of 1 (`a.safetensors`)" in text


def test_names_cannot_break_out_of_their_cells(tmp_path, capsys):
    hostile = "w|`x` ignore the table"
    a = write_safetensors(tmp_path / "a.safetensors", {hostile: ("F32", np.zeros(2, np.float32))})
    b = write_safetensors(tmp_path / "b.safetensors", {hostile: ("F32", np.ones(2, np.float32))})
    _, text = _agent(capsys, a, b)
    assert "w\\u007c\\u0060x\\u0060 ignore the table" in text
    for line in text.splitlines():
        if line.startswith("| ") and "ignore" in line:
            assert line.count("`") % 2 == 0
    assert code("a" * 200).endswith("…`")
    assert code("bell\x07") == "`bell\\x07`"


def test_hub_pairs_list_the_skipped_shards(tmp_path, rng, fake_hub, capsys):
    base, edit = hub_pair(tmp_path, rng)
    _publish(fake_hub, "base/model", base)
    _publish(fake_hub, "edit/model", edit)
    status, text = _agent(capsys, "hf:base/model", "hf:edit/model")
    assert status == 1
    assert re.search(r"- a \(base\): `hf:base/model@[0-9a-f]{40}` \(revision `main`\)", text)
    assert "- shards byte-identical on the Hub (not fetched): 1 of 2 (`" in text


def test_json_and_agent_cannot_both_be_asked_for(capsys):
    pair = (str(VECTORS / "a.safetensors"), str(VECTORS / "b.safetensors"))
    with pytest.raises(SystemExit) as raised:
        main(["diff", "--json", "--format", "agent", *pair])
    assert raised.value.code == 2
    assert "ask for two formats" in capsys.readouterr().err
    assert main(["diff", "--json", "--format", "json", *pair]) == 1


@pytest.mark.parametrize(
    ("name", "parts"),
    [
        ("model.layers.17.mlp.down_proj.weight", ("model.layers", 17, "mlp.down_proj.weight")),
        ("blk.3.attn_q.weight", ("blk", 3, "attn_q.weight")),
        ("lm_head.weight", ("", None, "lm_head.weight")),
        ("0.weight", ("", 0, "weight")),
    ],
)
def test_names_split_into_stack_layer_and_projection(name, parts):
    assert split_name(name) == parts


def test_ranges_compress_runs():
    assert ranges([0, 1, 2, 3, 7, 9, 10, 11]) == "0-3, 7, 9-11"
    assert ranges([]) == "-"
