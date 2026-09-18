import json

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS


def labels():
    frame = pd.DataFrame([np.linspace(i, 100, 11) for i in range(10)],
                         columns=CANONICAL_GRAIN_LABELS)
    frame.insert(0, 'sample_id', [f'SOIL_SECRET_{i}' for i in range(10)])
    return frame


def test_prompt_cannot_observe_query_label_or_soil_names():
    from soilgrain.visual_language import build_prompt
    truth = labels()
    prompt, examples = build_prompt(truth, 'SOIL_SECRET_3', 2)
    truth.loc[3, list(CANONICAL_GRAIN_LABELS)] = 100
    changed, changed_examples = build_prompt(truth, 'SOIL_SECRET_3', 2)
    assert prompt == changed
    assert examples == changed_examples
    assert len(examples) == len(set(examples)) == 8
    assert 'SOIL_SECRET_3' not in examples
    assert 'SOIL_SECRET_' not in prompt
    assert '2 query' in prompt


def test_invalid_curves_fail_instead_of_silently_repairing():
    from soilgrain.visual_language import parse_prediction
    valid = list(range(0, 101, 10))
    np.testing.assert_array_equal(parse_prediction(json.dumps({'cdf': valid})), valid)
    for bad in [valid[:-1], [0, 20, 10] + valid[3:], valid[:-1] + [99],
                [-1] + valid[1:], [float('nan')] + valid[1:]]:
        with pytest.raises(ValueError):
            parse_prediction(json.dumps({'cdf': bad}))
    with pytest.raises(ValueError):
        parse_prediction(json.dumps({'cdf': valid, 'extra': True}))


def test_tool_events_are_rejected_and_usage_is_required():
    from soilgrain.visual_language import audit_events
    events = [
        {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '{}'}},
        {'type': 'turn.completed', 'usage': {'input_tokens': 100, 'output_tokens': 20,
                                           'reasoning_output_tokens': 5}},
    ]
    assert audit_events(events) == {'input_tokens': 100, 'output_tokens': 20,
                                     'reasoning_output_tokens': 5}
    with pytest.raises(ValueError):
        audit_events(events[:-1])
    for kind in ['command_execution', 'mcp_tool_call', 'web_search', 'file_change']:
        with pytest.raises(ValueError):
            audit_events([{'type': 'item.completed', 'item': {'type': kind}}] + events)


def test_cli_disables_context_sources_and_tools(tmp_path):
    from soilgrain.visual_language import cli_command
    command = cli_command(tmp_path, [tmp_path / 'example_1.png'], tmp_path / 'schema.json')
    assert '--ephemeral' in command and '--ignore-user-config' in command
    assert '--ignore-rules' in command and 'project_doc_max_bytes=0' in command
    for feature in ['plugins', 'hooks', 'memories', 'multi_agent', 'shell_tool',
                    'unified_exec', 'view_image', 'apps', 'image_generation']:
        assert any(command[i:i+2] == ['--disable', feature] for i in range(len(command)-1))
    assert command[command.index('--cd') + 1] == str(tmp_path)


def test_crop_preserves_physical_extent_without_enlarging(tmp_path):
    from PIL import Image
    from soilgrain.visual_language import prepare_crop
    path = tmp_path / 'source.png'
    Image.new('RGB', (1000, 800), 'red').save(path)
    camera = pd.Series({'width': 2000, 'height': 1600, 'ppm': 10})
    output = tmp_path / 'anonymous.png'
    prepare_crop(path, camera, output)
    with Image.open(output) as image:
        assert image.size == (500, 500)
        assert image.getexif() == {}


def test_resume_checks_response_curve_prompt_and_image_integrity(tmp_path):
    from hashlib import sha256
    from soilgrain.visual_language import audit_cached_record
    response, prompt, image = [tmp_path / n for n in ('response.json', 'prompt.txt', 'image_01.png')]
    values = list(range(0, 101, 10))
    response.write_text(json.dumps({'cdf': values}))
    prompt.write_text('fixed prompt')
    image.write_bytes(b'image')
    record = {'folder': str(tmp_path), 'response_path': str(response), 'curve': values,
              'response_sha256': sha256(response.read_bytes()).hexdigest(),
              'prompt_sha256': sha256(prompt.read_bytes()).hexdigest(),
              'image_sha256': [sha256(image.read_bytes()).hexdigest()]}
    audit_cached_record(record)
    record['curve'] = [100] * 11
    with pytest.raises(ValueError):
        audit_cached_record(record)
    record['curve'] = values
    image.write_bytes(b'changed')
    with pytest.raises(ValueError):
        audit_cached_record(record)
