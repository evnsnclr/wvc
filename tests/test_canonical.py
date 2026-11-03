import hashlib

import numpy as np
import pytest

from wvc.canonical import _reject_floats, canonical_bytes, object_id

RECORD = {"time": "2026-11-02T00:00:00Z", "parents": [], "format": "wvc-commit/0", "author": "Évan"}
CANONICAL = '{"author":"Évan","format":"wvc-commit/0","parents":[],"time":"2026-11-02T00:00:00Z"}'
# Pinned so that any change to the canonical form shows up as a failing test.
RECORD_ID = "a79ad644ed330dc0f241d41b1408f47bbf260f76560e31c74628d60c37965943"


def test_keys_are_sorted_at_every_level():
    record = {"b": 1, "a": {"d": [3, {"z": 0, "y": None}], "c": True}}
    assert canonical_bytes(record) == b'{"a":{"c":true,"d":[3,{"y":null,"z":0}]},"b":1}'


def test_separators_have_no_whitespace():
    assert canonical_bytes({"a": [1, 2], "b": {"c": "d e"}}) == b'{"a":[1,2],"b":{"c":"d e"}}'


def test_non_ascii_is_utf8_not_escaped():
    data = canonical_bytes({"message": "naïve 模型 ✓"})
    assert data == '{"message":"naïve 模型 ✓"}'.encode()
    assert b"\\u" not in data


@pytest.mark.parametrize(
    "value", [1.0, float("nan"), np.float64(2.0), [1, 2.5], {"x": {"y": [0, 0.5]}}]
)
def test_floats_are_rejected_anywhere(value):
    with pytest.raises(ValueError, match="forbids floats"):
        canonical_bytes({"a": value})


def test_float_error_names_its_location():
    with pytest.raises(ValueError, match=r"\$\.a\.b\[1\]"):
        canonical_bytes({"a": {"b": [0, 1.5]}})


def test_the_float_check_walks_deep_nesting_without_recursion():
    """A deeply nested record gets ValueError, not RecursionError."""
    deep: list = [1.5]
    for _ in range(100_000):
        deep = [deep]
    with pytest.raises(ValueError, match="forbids floats"):
        _reject_floats(deep, "$")


def test_non_json_values_are_rejected():
    cycle: list = []
    cycle.append(cycle)
    for value in (b"bytes", {1, 2}, object(), np.int64(3), cycle):
        with pytest.raises(ValueError, match="not JSON-serialisable"):
            canonical_bytes({"a": value})
    with pytest.raises(ValueError, match="string keys"):
        canonical_bytes({"a": {1: "one"}})
    with pytest.raises(ValueError, match="UTF-8"):
        canonical_bytes({"a": "\ud800"})


def test_object_id_is_stable_across_key_order():
    reordered = dict(reversed(list(RECORD.items())))
    assert canonical_bytes(RECORD) == canonical_bytes(reordered) == CANONICAL.encode()
    assert object_id(RECORD) == object_id(reordered) == RECORD_ID
    assert hashlib.sha256(CANONICAL.encode()).hexdigest() == RECORD_ID
