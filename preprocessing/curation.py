"""Explicit reviewed intervals, never inferred automatically."""

from dataclasses import asdict
from pathlib import Path
import datetime
import hashlib
import json

import numpy as np
import yaml

from ..epic_cardio.confirmed_pulse import repair_confirmed_pulse
from .model import CURATED, OBSERVED


def load_curation(source):
    if source is None:
        return []
    if isinstance(source, (str, Path)):
        with Path(source).open() as stream:
            source = yaml.safe_load(stream)
    if isinstance(source, dict):
        source = source.get('entries', [source])
    entries = []
    for entry in source or []:
        entry = dict(entry)
        required = {'well', 'start_frame', 'end_frame', 'reviewer', 'date', 'reason'}
        if required - entry.keys():
            raise ValueError(f'Curation missing fields: {sorted(required - entry.keys())}')
        if any(not entry[key] for key in ('well', 'reviewer', 'date', 'reason')):
            raise ValueError('Curation requires well, reviewer, date and reason')
        if isinstance(entry['date'], (datetime.date, datetime.datetime)):
            entry['date'] = entry['date'].isoformat()
        if not isinstance(entry['start_frame'], int) or not isinstance(entry['end_frame'], int):
            raise ValueError('Curation frame bounds must be integers')
        entries.append(entry)
    return entries


def curation_hash(curation):
    payload = json.dumps(load_curation(curation), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def apply_curation(state, config, curation):
    result = state.copy()
    receipts = []
    # Every interval uses original observed evidence, never another repair's anchors.
    evidence = state.values if state.acquired_values is None else state.acquired_values
    valid = state.method == OBSERVED if state.acquired_valid is None else state.acquired_valid
    for entry in load_curation(curation):
        if entry['well'] != state.raw.well:
            continue
        settings = asdict(config.curation)
        settings['clean_frames'] = entry.get('clean_frames', settings['clean_frames'])
        settings['threshold_pm'] = entry.get('threshold_pm', entry.get('threshold', settings['threshold_pm']))
        start, end = entry['start_frame'], entry['end_frame']
        values, accepted, receipt = repair_confirmed_pulse(
            evidence, valid, state.raw.time_s, state.raw.phases, state.allowed, start, end, **settings)
        # Detection uses original observations. Correction honors any earlier stitched baseline.
        anchors = np.isfinite(state.values[start - 1]) & np.isfinite(state.values[end])
        accepted &= anchors
        fraction = ((state.raw.time_s[start:end] - state.raw.time_s[start - 1])
                    / (state.raw.time_s[end] - state.raw.time_s[start - 1]))
        interpolation = (state.values[start - 1][None]
                         + fraction[:, None, None] * (state.values[end] - state.values[start - 1])[None])
        result.values[start:end, accepted] = interpolation[:, accepted]
        result.method[start:end, accepted] = CURATED
        receipt['repaired_pixels'] = int(accepted.sum())
        receipts.append(dict(entry=entry, parameters=settings, **receipt))
    result.receipts['curated_repairs'] = receipts
    return result
