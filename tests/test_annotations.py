"""Notation syntax, semantic identities and stable session payload validation."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('well_annotations', Path(__file__).parents[1] / 'annotations.py')
annotations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annotations)
parse = annotations.parse_annotation


@pytest.mark.parametrize('text,cell,donor', [
    ('Noncoated, HeLa -> SAL1(1.5 uM)', 'hela', None),
    ('Gelatin, Preo -> HeLa-EV', 'preo', 'hela'),
    ('RGD(50%), HeLa -> Preo-EV(dose=?)', 'hela', 'preo'),
])
def test_recipient_donor_and_unknown_dose_remain_distinct(text, cell, donor):
    result = parse(text)
    assert result['phases'][0]['actions'][0]['identity'] == cell
    if donor:
        assert result['phases'][1]['actions'][0]['donor'] == donor
    else:
        assert result['phases'][1]['actions'][0]['parameters']['final'] == dict(value=1.5, unit='uM')


def test_phase_boundaries_simultaneous_additions_no_change_and_order():
    result = parse('Noncoated, NK92 -> HeLa + SAL1(5 µM) -> -')
    assert len(result['phases']) == 3
    assert len(result['phases'][1]['actions']) == 2
    assert result['phases'][0]['actions'][0]['identity'] == 'nk92'
    assert result['phases'][2]['actions'] == [dict(kind='no_change')]


def test_unknown_coating_is_not_noncoated_and_concentrations_have_basis():
    assert parse('Unknown, DMEM')['coating']['type'] is None
    assert parse('Noncoated, Buffer')['coating']['type'] == 'noncoated'
    result = parse('Gelatin(0.2%), HeLa(density=12000 cells/mL) -> SAL1(stock=30 mM, volume=20 uL)')
    params = result['phases'][1]['actions'][0]['parameters']
    assert 'final' not in params
    assert params['stock']['value'] == 30
    assert result['coating']['concentration'] == dict(value=.2, unit='%')


@pytest.mark.parametrize('text', [
    '', 'HeLa + SAL1', 'Noncoated, HeLa ->', 'Noncoated, HeLa -> - + SAL1(5 uM)',
    'RGD(101%), HeLa', 'Noncoated(5%), HeLa', 'RGD(10 uM), HeLa',
    'Gelatin, HeLa -> SAL1(-1 uM)', 'Gelatin, HeLa -> SAL1(5 frogs)',
    'Gelatin, HeLa -> SAL1(final=5 uM, final=3 uM)', 'Gelatin, HeLa -> SAL1(volume=5 uM)',
    'Noncoated, HeLa(density=4 uM)', 'Noncoated, HeLa -> Unrecognized',
    'Noncoated, -', 'Noncoated, a: HeLa -> a: -', 'Noncoated, HeLa -> SAL1(5 uM',
])
def test_invalid_or_ambiguous_annotations_are_rejected(text):
    with pytest.raises(ValueError):
        parse(text)


def test_named_phases_washes_and_replacement_are_explicit():
    result = parse('RGD, baseline: Buffer -> adhesion: HeLa -> wash: Wash(DMEM) -> fresh: Replace(DMEM)')
    assert [p['label'] for p in result['phases']] == ['baseline', 'adhesion', 'wash', 'fresh']
    assert [p['actions'][0]['kind'] for p in result['phases']] == ['vehicle', 'cells', 'wash', 'replace']


def test_plate_roundtrip_and_tamper_detection():
    payload = annotations.build_plate_annotations({'A1': 'Noncoated, HeLa -> NK92 -> -'},
                                                  phase_binding=[1, 2, 3], valid_wells=['A1'])
    assert annotations.validate_plate_annotations(payload, valid_wells=['A1']) == payload
    changed = deepcopy(payload)
    changed['wells']['A1']['coating']['type'] = 'gelatin'
    with pytest.raises(ValueError, match='disagree'):
        annotations.validate_plate_annotations(changed)
    assert payload['wells']['A1']['phase_binding'] == [1, 2, 3]


@pytest.mark.parametrize('binding', [[0], [1, 1], [-1, 0], [1, 0], [True, 2]])
def test_invalid_recorded_phase_binding_rejected(binding):
    with pytest.raises(ValueError, match='binding'):
        annotations.build_plate_annotations({'A1': 'RGD, HeLa -> -'}, phase_binding=binding)
