import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soilgrain.constants import CANONICAL_GRAIN_LABELS


def label_frame(ids):
    frame = pd.DataFrame([np.linspace(i, 100, 11) for i in range(len(ids))],
                         columns=CANONICAL_GRAIN_LABELS)
    frame.insert(0, 'sample_id', ids)
    return frame


def test_retrieval_uses_soil_means_training_population_std_and_stable_ties():
    from soilgrain.retrieval_visual_language import select_examples
    ids = [f'S{i:02d}' for i in range(10)]
    rows = []
    for i, sample_id in enumerate(ids + ['query', 'irrelevant_test']):
        # Two photographs each; within-soil deviations must not enter the scaler.
        for offset in (-100, 100):
            value = i if i < 10 else (4.5 if sample_id == 'query' else 1e9)
            rows.append({'sample_id': sample_id, 'split': 'train' if i < 10 else 'test',
                         **{f'feature_{j}': value + offset if j == 0 else 7 for j in range(23)}})
    photos = pd.DataFrame(rows).sample(frac=1, random_state=12)
    result = select_examples(photos, ids[::-1], 'query')
    assert result['examples'] == ['S04', 'S05', 'S03', 'S06', 'S02', 'S07', 'S01', 'S08']
    assert result['scaler_training_ids'] == ids
    assert result['scaler_mean'] == [4.5] + [7] * 22
    assert result['scaler_std'][0] == pytest.approx(np.sqrt(8.25))
    assert result['scaler_std'][1:] == [1] * 22
    assert result['distances'] == pytest.approx([.25 / 8.25] * 2 + [2.25 / 8.25] * 2
                                               + [6.25 / 8.25] * 2 + [12.25 / 8.25] * 2)
    heldout = select_examples(photos, ids, 'S09')
    assert heldout['scaler_training_ids'] == ids[:-1]
    assert heldout['scaler_mean'][0] == 4
    assert heldout['scaler_std'][0] == pytest.approx(np.sqrt(60 / 9))
    assert 'S09' not in heldout['examples']
    photos.loc[photos.sample_id == 'S09', 'feature_0'] = 1e8
    changed = select_examples(photos, ids, 'S09')
    assert changed['scaler_mean'] == heldout['scaler_mean']
    assert changed['scaler_std'] == heldout['scaler_std']


def test_retrieval_prompt_changes_only_example_lines_and_cannot_observe_query_label():
    from soilgrain.retrieval_visual_language import build_retrieval_prompt
    from soilgrain.visual_language import build_prompt
    truth = label_frame([f'SECRET_{i}' for i in range(10)])
    examples = [f'SECRET_{i}' for i in range(8, 0, -1)]
    prompt = build_retrieval_prompt(truth, 'SECRET_0', 2, examples)
    original, _ = build_prompt(truth, 'SECRET_0', 2)
    assert prompt.splitlines()[0] == original.splitlines()[0]
    assert prompt.splitlines()[-1] == original.splitlines()[-1]
    assert json.loads(prompt.splitlines()[1].split(': ', 1)[1])[0] == 8
    assert 'SECRET_' not in prompt
    truth.loc[0, list(CANONICAL_GRAIN_LABELS)] = 100
    assert build_retrieval_prompt(truth, 'SECRET_0', 2, examples) == prompt
    with pytest.raises(ValueError):
        build_retrieval_prompt(truth, 'SECRET_0', 2, ['SECRET_0'] + examples[:-1])


@pytest.fixture
def assay(tmp_path, monkeypatch):
    import soilgrain.retrieval_visual_language as runner
    truth = label_frame([f'S{i:02d}' for i in range(24)])
    sample = label_frame([f'T{i:02d}' for i in range(10)])
    rows = []
    for i, sample_id in enumerate(truth.sample_id.tolist() + sample.sample_id.tolist()):
        path = tmp_path / f'{sample_id}.png'
        Image.new('RGB', (100, 100), (i, 0, 0)).save(path)
        rows.append({'sample_id': sample_id, 'split': 'train' if i < 24 else 'test',
                     'camera': 'camera', 'path': str(path),
                     **{f'feature_{j}': i if j == 0 else 0 for j in range(23)}})
    photos = pd.DataFrame(rows)
    cameras = pd.DataFrame([{'phone': 'camera', 'width': 100, 'height': 100, 'ppm': 1}]).set_index('phone')
    source = tmp_path / 'source.txt'
    source.write_text('fixed source')
    sources = [{'path': str(source), 'sha256': sha256(source.read_bytes()).hexdigest()}]
    cfg = SimpleNamespace(artifacts_dir=tmp_path)
    monkeypatch.setattr(runner, 'load_inputs', lambda _: (cfg, truth, sample, photos, cameras, sources))
    monkeypatch.setattr(runner, 'runtime_identity', lambda: {
        'client_version': 'codex-cli test', 'producing_commit': 'fixture',
        'code_sources': [{'path': str(source), 'sha256': sha256(source.read_bytes()).hexdigest()}],
    })
    output = tmp_path / 'experiments' / 'september18_round3' / 'retrieval_vlm'
    state = SimpleNamespace(calls=0, mode='ok', input_tokens=100, output_tokens=20)

    def request(command, *, input, text, stdout, stderr, timeout):
        state.calls += 1
        assert command[0] == '/opt/homebrew/bin/codex'
        assert '--ignore-rules' in command and '--ephemeral' in command
        assert 'SECRET' not in input and 'scaler' not in input
        folder = Path(command[command.index('--cd') + 1])
        assert all(not p.name.startswith(('S', 'T')) for p in folder.iterdir())
        response = json.dumps({'cdf': list(range(0, 101, 10))})
        if state.mode == 'malformed':
            response = '{"cdf": [1, 0]}'
        (folder / 'response.json').write_text(response)
        events = [
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': response}},
            {'type': 'turn.completed', 'usage': {'input_tokens': state.input_tokens,
                                               'output_tokens': state.output_tokens,
                                               'reasoning_output_tokens': 5}},
        ]
        if state.mode == 'tool':
            events.insert(0, {'type': 'item.completed', 'item': {'type': 'command_execution'}})
        if state.mode == 'wrong_final':
            events[0]['item']['text'] = json.dumps({'cdf': [100] * 11})
        stdout.write('\n'.join(json.dumps(e) for e in events))
        if state.mode == 'timeout':
            raise runner.subprocess.TimeoutExpired(command, timeout)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, 'run', request)
    return SimpleNamespace(run=runner.run_retrieval_visual_language, output=output, state=state,
                           truth=truth, photos=photos, cameras=cameras, source=source,
                           sources=sources, runner=runner)


def test_full_assay_requires_34_responses_and_resume_preserves_all_artifacts(assay):
    summary = assay.run()
    assert summary['experiment'] == 'retrieval_visual_language'
    assert summary['eligible'] is True
    assert summary['heldout_soils'] == 24 and summary['test_soils'] == 10
    assert summary['requests'] == 34 and summary['output_tokens'] == 680
    assert len(pd.read_csv(summary['oof']['path'])) == 24
    assert len(pd.read_csv(summary['submission']['path'])) == 10
    ledger = json.loads((assay.output / 'requests.json').read_text())
    assert ledger['requests'][0]['sample_id'] not in ledger['requests'][0]['examples']
    assert len(ledger['requests'][0]['scaler_training_ids']) == 23
    assert len(ledger['requests'][-1]['scaler_training_ids']) == 24
    paths = list(assay.output.rglob('*'))
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in paths if p.is_file()}
    assert assay.run() == summary
    assert assay.state.calls == 34
    after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in paths if p.is_file()}
    assert after == before


@pytest.mark.parametrize('mode', ['malformed', 'tool', 'wrong_final', 'timeout'])
def test_failed_or_uncertain_requests_stop_without_retry(assay, mode):
    assay.state.mode = mode
    with pytest.raises((ValueError, assay.runner.subprocess.TimeoutExpired)):
        assay.run()
    assert assay.state.calls == 1
    ledger = json.loads((assay.output / 'requests.json').read_text())
    assert ledger['requests'][0]['state'] == 'failed'
    if mode in ('malformed', 'wrong_final'):
        assert ledger['requests'][0]['output_tokens'] == 20
    assert (assay.output / 'failure.json').is_file()
    assert not (assay.output / 'summary.json').exists()
    with pytest.raises(ValueError, match='retry'):
        assay.run()
    assert assay.state.calls == 1


@pytest.mark.parametrize('token_kind,amount', [('input_tokens', 1_000_000), ('output_tokens', 50_000)])
def test_aggregate_budget_stops_before_next_request(assay, token_kind, amount):
    setattr(assay.state, token_kind, amount)
    with pytest.raises(ValueError, match='budget'):
        assay.run()
    assert assay.state.calls == 1
    assert not (assay.output / 'summary.json').exists()


@pytest.mark.parametrize('change', ['source', 'events', 'prompt', 'image', 'client', 'code', 'examples'])
def test_resume_rejects_changed_provenance_or_completed_context(assay, change, monkeypatch):
    assay.run()
    ledger_path = assay.output / 'requests.json'
    ledger = json.loads(ledger_path.read_text())
    record = ledger['requests'][0]
    folder = Path(record['folder'])
    if change == 'source':
        assay.source.write_text('changed source')
    elif change == 'events':
        (folder / 'events.jsonl').write_text('{}')
    elif change == 'prompt':
        (folder / 'prompt.txt').write_text('tampered')
        record['prompt_sha256'] = sha256(b'tampered').hexdigest()
    elif change == 'image':
        Image.new('RGB', (100, 100), 'yellow').save(folder / 'image_01.png')
        record['image_sha256'][0] = sha256((folder / 'image_01.png').read_bytes()).hexdigest()
        for source in record['raw_files']:
            if source['path'] == str(folder / 'image_01.png'):
                source['sha256'] = record['image_sha256'][0]
    elif change == 'client':
        ledger['client_version'] = 'another client'
    elif change == 'code':
        ledger['code_sources'] = []
    else:
        record['examples'].reverse()
    ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError):
        assay.run()
    assert assay.state.calls == 34


def test_completed_prefix_resumes_without_repeating_model_requests(assay, monkeypatch):
    def interrupt_after_completed(*args, **kwargs):
        raise KeyboardInterrupt('operator stopped between requests')

    monkeypatch.setattr(assay.runner, 'print', interrupt_after_completed, raising=False)
    with pytest.raises(KeyboardInterrupt):
        assay.run()
    ledger = json.loads((assay.output / 'requests.json').read_text())
    assert len(ledger['requests']) == 1 and ledger['requests'][0]['state'] == 'complete'
    monkeypatch.setattr(assay.runner, 'print', lambda *args, **kwargs: None)
    summary = assay.run()
    assert summary['eligible'] and assay.state.calls == 34
