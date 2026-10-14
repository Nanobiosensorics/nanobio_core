"""Versioned, immutable settings for quantitative preprocessing."""

from dataclasses import asdict, dataclass, field, fields, replace
import hashlib
import json

from ..epic_cardio.pixel_recovery import RecoveryConfig as OriginalRecoveryConfig


@dataclass(frozen=True)
class ValidityConfig:
    max_jump_pm: float = 1000.0
    flat_trace_is_missing: bool = True


@dataclass(frozen=True)
class ActiveAreaConfig:
    margin_px: int = 2


@dataclass(frozen=True)
class RegistrationConfig:
    enabled: bool = True
    max_shift_px: int = 3
    threshold_pm: float = 75.0
    neighborhood_size: int = 3


@dataclass(frozen=True)
class RecoveryConfig(OriginalRecoveryConfig):
    enabled: bool = True
    anchor_frames: int = 3
    step_window: int = 6
    step_history: int = 12
    median_radius: int = 5
    minimum_observations: int = 3
    noise_scale: float = 1.4826
    sign_agreement: float = 0.8
    minimum_displacement: float = 0.3
    neighbor_radius: int = 1
    neighbor_tolerance_pm: float = 25.0
    neighbor_tolerance_fraction: float = 0.25
    spatial_cache_size: int = 8192
    donor_weight_epsilon: float = 1e-12


@dataclass(frozen=True)
class CurationConfig:
    clean_frames: int = 6
    threshold_pm: float = 75.0
    minimum_neighbors: int = 2
    noise_multiplier: float = 6.0
    noise_scale: float = 1.4826
    minimum_clean_frames: int = 3
    neighbor_radius: int = 1


@dataclass(frozen=True)
class ReferenceConfig:
    frames: int = 5


@dataclass(frozen=True)
class BackgroundConfig:
    method: str = 'median'
    n_pixels: int = 7
    spacing: int = 2
    threshold_pm: float = 75.0
    coordinates: tuple = ()


@dataclass(frozen=True)
class PhaseOffsetConfig:
    mode: str = 'robust'
    anchor_frames: int = 5


@dataclass(frozen=True)
class ExportConfig:
    min_observed_weight: float = 0.5


@dataclass(frozen=True)
class LegacyConfig:
    pixel_filtering: bool = True
    frame_jump_correction: bool = True
    frame_jump_threshold: float = 75.0
    phase_offset_correction: bool = True
    random_seed: int = 0


@dataclass(frozen=True)
class PreprocessConfig:
    version: int = 1
    profile_name: str = 'default'
    validity: ValidityConfig = field(default_factory=ValidityConfig)
    active_area: ActiveAreaConfig = field(default_factory=ActiveAreaConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    phase_offset: PhaseOffsetConfig = field(default_factory=PhaseOffsetConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    legacy: LegacyConfig = field(default_factory=LegacyConfig)

    def __post_init__(self):
        if self.version != 1:
            raise ValueError('Unsupported preprocessing config version')
        if self.profile_name not in ('default', 'legacy', 'observed_only'):
            raise ValueError('Unknown preprocessing profile')
        if self.phase_offset.mode not in ('none', 'single', 'robust'):
            raise ValueError('Unknown phase offset mode')
        if self.background.method not in ('none', 'mean', 'median'):
            raise ValueError('Unknown background method')
        if self.reference.frames < 1 or self.phase_offset.anchor_frames < 2:
            raise ValueError('Reference and fit anchor counts must be positive (fit requires two)')
        if not 0 <= self.export.min_observed_weight <= 1:
            raise ValueError('Export weight must be between zero and one')
        if self.active_area.margin_px not in (1, 2):
            raise ValueError('Active-area margin must be one or two pixels')
        recovery = self.recovery
        positive = (recovery.temporal_gap, recovery.anchor_frames, recovery.step_window,
                    recovery.minimum_observations, recovery.tail_frames, recovery.max_cluster_area,
                    recovery.spatial_cache_size, self.background.n_pixels)
        if min(positive) < 1 or not recovery.radii or min(recovery.radii) < 1:
            raise ValueError('Recovery and background limits must be positive')
        if recovery.step_window < 2 or recovery.step_history <= recovery.step_window:
            raise ValueError('Step history must extend beyond the fit window (at least two frames)')
        if (recovery.threshold_pm <= 0 or recovery.noise_multiplier <= 0
                or self.validity.max_jump_pm <= 0 or recovery.median_radius < 1):
            raise ValueError('Validity and recovery thresholds must be positive')

    @classmethod
    def profile(cls, name):
        name = name.replace('-', '_')
        if name not in ('default', 'legacy', 'observed_only'):
            raise ValueError(f'Unknown preprocessing profile: {name}')
        config = cls(profile_name=name)
        if name == 'observed_only':
            config = replace(config, recovery=replace(config.recovery, enabled=False))
        if name == 'legacy':
            config = replace(config, reference=ReferenceConfig(1),
                             background=replace(config.background, method='mean'),
                             phase_offset=PhaseOffsetConfig('single'))
        return config

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, data):
        defaults = cls()
        known = {item.name for item in fields(cls)}
        if set(data) - known:
            raise ValueError(f'Unknown settings: {sorted(set(data) - known)}')
        values = dict(data)
        for item in fields(cls):
            default = getattr(defaults, item.name)
            if hasattr(default, '__dataclass_fields__') and item.name in values:
                section = dict(values[item.name])
                for key in ('radii', 'coordinates'):
                    if key in section:
                        section[key] = tuple(tuple(v) if isinstance(v, list) else v for v in section[key])
                values[item.name] = type(default)(**section)
        return cls(**values)

    def config_hash(self):
        return config_hash(self)


def config_hash(config):
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()
