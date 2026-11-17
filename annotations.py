"""Versioned, dependency-free well treatment annotations.

Notation describes interventions at phase boundaries, never measured times or outcomes.
The parser accepts only explicit syntax and never interprets legacy free-text labels.
"""

import math
import re
from copy import deepcopy


SCHEMA_VERSION = 1
COATINGS = {'noncoated': 'noncoated', 'rgd': 'ppr', 'ppr': 'ppr',
            'gelatin': 'gelatin', 'fibronectin': 'fibronectin', 'unknown': None}
CELLS = {'hela': 'hela', 'preo': 'preo', 'mc3t3-e1': 'preo', 'nk92': 'nk92'}
UNITS = {'uM', 'mM', 'nM', 'uL', 'mL', 'mg/mL', 'ug/mL', 'cells/mL', 'particles/mL', '%'}


def split_terms(value, separator):
    """Split separators outside parentheses and reject unmatched parentheses."""
    result, depth, start = [], 0, 0
    for index, character in enumerate(value):
        if character == '(':
            depth += 1
        elif character == ')':
            depth -= 1
            if depth < 0:
                raise ValueError('Unbalanced annotation parentheses')
        if character == separator and depth == 0:
            result.append(value[start:index].strip())
            start = index + 1
    if depth:
        raise ValueError('Unbalanced annotation parentheses')
    result.append(value[start:].strip())
    if any(not term for term in result):
        raise ValueError('Empty annotation term')
    return result


def quantity(value):
    value = value.strip().replace('µ', 'u').replace('μ', 'u')
    if value == '?':
        return None
    match = re.fullmatch(r'(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([A-Za-z/%]+)', value)
    if not match or match[2] not in UNITS:
        raise ValueError(f'Invalid quantity or unit: {value!r}')
    number = float(match[1])
    if not math.isfinite(number) or number <= 0:
        raise ValueError('Annotation quantities must be finite and positive')
    if match[2] == '%' and number > 100:
        raise ValueError('Percentage exceeds 100')
    return dict(value=number, unit=match[2])


def _token(value):
    match = re.fullmatch(r'([A-Za-z][A-Za-z0-9_-]*)(?:\(([^()]*)\))?', value.strip())
    if not match:
        raise ValueError(f'Invalid annotation token: {value!r}')
    return match[1], match[2]


def _parameters(body, shorthand):
    if body is None:
        return {}
    result = {}
    for term in split_terms(body, ','):
        key, value = term.split('=', 1) if '=' in term else (shorthand, term)
        key = key.strip()
        if key not in {'final', 'stock', 'added', 'dose', 'density', 'volume', 'concentration'}:
            raise ValueError(f'Unknown annotation parameter: {key}')
        if key in result:
            raise ValueError(f'Duplicate annotation parameter: {key}')
        parsed = quantity(value)
        if parsed is not None:
            unit = parsed['unit']
            if key == 'volume' and unit not in {'uL', 'mL'}:
                raise ValueError('Volume requires uL or mL')
            if key == 'density' and unit != 'cells/mL':
                raise ValueError('Cell suspension density requires cells/mL')
            if key in {'final', 'stock', 'added'} and unit not in {'uM', 'mM', 'nM', 'mg/mL', 'ug/mL'}:
                raise ValueError('Concentration requires concentration units')
        result[key] = parsed
    return result


def _action(term):
    if term == '-':
        return dict(kind='no_change')
    if term.casefold() == 'unspecified':
        return dict(kind='unspecified')
    operation = re.fullmatch(r'(Wash|Replace)\((DMEM|Buffer)\)', term, re.IGNORECASE)
    if operation:
        return dict(kind=operation[1].lower(), identity=operation[2].lower())
    name, body = _token(term)
    lower = name.casefold()
    if lower in CELLS:
        return dict(kind='cells', identity=CELLS[lower], parameters=_parameters(body, 'density'))
    if lower.endswith('-ev') and lower[:-3] in CELLS:
        return dict(kind='vesicles', donor=CELLS[lower[:-3]], parameters=_parameters(body, 'dose'))
    if lower in {'sal1', 'dmem', 'buffer'}:
        kind = 'compound' if lower == 'sal1' else 'vehicle'
        return dict(kind=kind, identity=lower, parameters=_parameters(body, 'final'))
    raise ValueError(f'Unknown annotation material: {name!r}')


def parse_annotation(notation):
    """Return canonical JSON-serializable coating and ordered phase actions."""
    if not isinstance(notation, str) or not notation.strip():
        raise ValueError('A nonempty well annotation is required')
    if len(notation) > 4096 or any(ord(character) < 32 for character in notation):
        raise ValueError('Annotation is too long or contains control characters')
    text = notation.strip().replace('→', '->')
    raw_phases = text.split('->')
    initial = split_terms(raw_phases[0], ',')
    if len(initial) != 2:
        raise ValueError('Expected "Coating, Initial materials" before the first arrow')
    name, body = _token(initial[0])
    if name.casefold() not in COATINGS:
        raise ValueError(f'Unknown coating: {name}')
    coating = dict(type=COATINGS[name.casefold()], concentration=None, mixture_percent=None)
    if body is not None:
        amount = quantity(body)
        if coating['type'] in {None, 'noncoated'}:
            raise ValueError('Unknown/noncoated surfaces cannot declare a concentration')
        if coating['type'] == 'ppr':
            if amount is None or amount['unit'] != '%':
                raise ValueError('RGD mixture fraction requires percent')
            coating['mixture_percent'] = amount['value']
        else:
            coating['concentration'] = amount
    phases = []
    names = set()
    for index, raw in enumerate([initial[1], *raw_phases[1:]]):
        raw = raw.strip()
        label = None
        if ':' in raw:
            label, raw = [part.strip() for part in raw.split(':', 1)]
            if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', label) or label in names:
                raise ValueError('Phase labels must be unique identifiers')
            names.add(label)
        actions = [_action(term) for term in split_terms(raw, '+')]
        if len(actions) > 1 and any(a['kind'] in {'no_change', 'unspecified', 'wash', 'replace'} for a in actions):
            raise ValueError('No-change/unspecified/wash/replacement must be the only phase action')
        if index == 0 and any(a['kind'] in {'no_change', 'wash', 'replace'} for a in actions):
            raise ValueError('Initial phase must establish materials')
        phases.append(dict(index=index, label=label, actions=actions))
    return dict(schema_version=SCHEMA_VERSION, notation=text, coating=coating, phases=phases)


def build_plate_annotations(assignments, *, phase_binding=None, source=None, valid_wells=None):
    """Validate a complete import before returning an atomic session payload.

    Phase bindings map displayed phases to recorded indices; they never infer clocks.
    A binding may be omitted while authoring, but consumers may require one.
    """
    known = set(valid_wells) if valid_wells is not None else None
    wells = {}
    for well, notation in assignments.items():
        if not isinstance(well, str) or not re.fullmatch(r'[A-Z]+[1-9]\d*', well):
            raise ValueError(f'Invalid well identity: {well!r}')
        if known is not None and well not in known:
            raise ValueError(f'Unknown well: {well}')
        parsed = parse_annotation(notation)
        binding = phase_binding.get(well) if isinstance(phase_binding, dict) else phase_binding
        if binding is not None:
            if (not isinstance(binding, list) or len(binding) != len(parsed['phases'])
                    or any(type(i) is not int or i < 0 for i in binding)
                    or binding != sorted(set(binding))):
                raise ValueError(f'{well}: phase binding must match phases and strictly increase')
        wells[well] = dict(parsed, phase_binding=deepcopy(binding))
    if not wells:
        raise ValueError('Plate annotations cannot be empty')
    return dict(schema_version=SCHEMA_VERSION, wells=wells, source=deepcopy(source))


def validate_plate_annotations(payload, *, valid_wells=None):
    """Reparse stored text and reject stale/tampered structured fields on reload."""
    if not isinstance(payload, dict) or payload.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported plate annotation schema version')
    try:
        assignments = {well: item['notation'] for well, item in payload['wells'].items()}
        bindings = {well: item.get('phase_binding') for well, item in payload['wells'].items()}
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Invalid stored plate annotations') from error
    normalized = build_plate_annotations(assignments, phase_binding=bindings,
                                         source=payload.get('source'), valid_wells=valid_wells)
    if payload != normalized:
        raise ValueError('Structured plate annotation fields disagree with their notation')
    return normalized
