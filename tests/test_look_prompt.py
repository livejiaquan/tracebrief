"""The device path's look-v2 prompt and parser are the frozen ones in scripts/look_local.py.

look_local.py is read as source (ast), never imported, so llama_cpp is not needed.
"""
import ast
import hashlib
import inspect
from pathlib import Path

import pytest

from tracebrief import look_prompt

ROOT = Path(__file__).resolve().parents[1]


def frozen():
    tree = ast.parse((ROOT / "scripts" / "look_local.py").read_text(encoding="utf-8"))
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in {"PROMPT", "PROMPT_VERSION", "YES_NO"}}
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    return constants, functions


def test_prompt_is_byte_identical_to_the_frozen_look_v2_prompt():
    constants, _ = frozen()
    assert constants["PROMPT_VERSION"] == look_prompt.PROMPT_VERSION == "look-v2"
    assert constants["PROMPT"].encode() == look_prompt.PROMPT.encode()
    assert constants["YES_NO"] == look_prompt.YES_NO
    # The CPU record's prompt_sha256 (look_local.py) is therefore also the device record's.
    assert hashlib.sha256(constants["PROMPT"].encode()).hexdigest() == hashlib.sha256(look_prompt.PROMPT.encode()).hexdigest()


def test_parser_is_the_frozen_parser():
    _, functions = frozen()
    ours = ast.parse(inspect.getsource(look_prompt.parse)).body[0]
    assert ast.dump(ours) == ast.dump(functions["parse"])


def test_parser_reads_the_json_object_inside_model_text():
    text = 'Sure.\n```json\n{"forklift": "Yes", "person": " no ", "description": "A forklift waits in the aisle."}\n```'
    assert look_prompt.parse(text) == {"forklift": "yes", "person": "no", "description": "A forklift waits in the aisle."}


def test_parser_keeps_at_most_40_words():
    words = " ".join(f"w{i}" for i in range(60))
    answer = look_prompt.parse('{"forklift": "yes", "person": "yes", "description": "%s"}' % words)
    assert answer["description"].split() == [f"w{i}" for i in range(40)]


@pytest.mark.parametrize("text, message", [
    ("I cannot see an image.", "No JSON"),
    ('{"forklift": "maybe", "person": "yes", "description": "x"}', "forklift"),
    ('{"forklift": "yes", "description": "x"}', "person"),
    ('{"forklift": "yes", "person": "yes", "description": "  "}', "description"),
])
def test_parser_rejects_unusable_answers(text, message):
    with pytest.raises(ValueError, match=message):
        look_prompt.parse(text)
