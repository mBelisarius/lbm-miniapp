"""Expands a campaign description into the list of runs it requires.

A study declares lists of parameters and the plan is their cartesian product.
Each run is a plain dictionary, so it can be stored in the ledger and read back
by the report stage without any conversion.
"""
import hashlib
import itertools
import json

from common import as_list


def study_value(study, defaults, key, fallback_key=None):
    """Reads a parameter of a study, falling back to the campaign defaults."""
    if key in study:
        return study[key]
    if fallback_key is not None and fallback_key in study:
        return study[fallback_key]
    return defaults.get(key)


def run_id(run):
    """Builds a readable and unique identifier for a run."""
    label = (
        f"{run['backend']}_{run['target']}"
        f"_nx{run['nx']}_ny{run['ny']}"
        f"_plpr{run['plpr']:g}_nu{run['viscosity']:g}"
        f"_c{run['cores']}_r{run['repeat']}"
    )
    signature = json.dumps({k: run[k] for k in sorted(run) if k != 'timeout'}, sort_keys=True)
    digest = hashlib.sha1(signature.encode('utf-8')).hexdigest()[:8]
    return f"{label}_{digest}"


def run_key(run):
    return f"{run['study']}/{run_id(run)}"


def build_run_config(base_config, run):
    """Derives the YAML configuration of a single run from the base config."""
    config = {key: dict(value) if isinstance(value, dict) else value
              for key, value in base_config.items()}

    fluid = config.setdefault('Fluid', {})
    mesh = config.setdefault('Mesh', {})
    control = config.setdefault('Control', {})
    performance = config.setdefault('Performance', {})

    pressure_l = float(fluid.get('pressureL', 1.0))
    pressure_r_base = float(fluid.get('pressureR', 0.1))
    density_r_base = float(fluid.get('densityR', 0.125))

    # The left state is kept fixed and the right state follows the isentropic
    # relation pL / rhoL^gamma = pR / rhoR^gamma.
    pressure_r = pressure_l / run['plpr']
    density_r = density_r_base * (pressure_r / pressure_r_base) ** (1.0 / run['gamma'])

    fluid['pressureR'] = float(pressure_r)
    fluid['densityR'] = float(density_r)
    fluid['viscosity'] = float(run['viscosity'])
    fluid['prandtl'] = float(run['prandtl'])
    fluid['gamma'] = float(run['gamma'])

    mesh['nx'] = int(run['nx'])
    mesh['ny'] = int(run['ny'])

    control['tmax'] = float(run['tmax'])
    control['printStep'] = int(run['printStep'])

    performance['target'] = run['target']
    performance['backend'] = run['backend']
    performance['cores'] = int(run['cores'])

    return config


class CampaignPlan:
    """List of runs described by a campaign file."""

    def __init__(self, campaign, overrides=None):
        self.defaults = dict(campaign.get('defaults', {}) or {})
        self.defaults.update({key: value for key, value in (overrides or {}).items()
                              if value is not None})
        self.studies = campaign.get('studies', {}) or {}
        self.overrides = {key: value for key, value in (overrides or {}).items()
                          if value is not None}

    def runs(self, studies=None):
        """Expands every selected study into the runs it requires."""
        runs = []
        for name, study in self.studies.items():
            if studies and name not in studies:
                continue
            runs.extend(self._expand(name, study or {}))
        return runs

    def _expand(self, name, study):
        study = dict(study)
        study.update(self.overrides)
        defaults = self.defaults

        kind = study.get('kind', 'accuracy')
        backends = as_list(study_value(study, defaults, 'backends', 'backend')) or ['Plain']
        targets = as_list(study_value(study, defaults, 'targets', 'target')) or ['CPU']
        nxs = as_list(study_value(study, defaults, 'nx'))
        plprs = as_list(study_value(study, defaults, 'plpr')) or [10.0]
        viscosities = as_list(study_value(study, defaults, 'viscosity'))
        cores_list = as_list(study_value(study, defaults, 'cores')) or [0]
        repeats = int(study_value(study, defaults, 'repeats') or 1)
        warmup = bool(study_value(study, defaults, 'warmup'))
        timeout = float(study_value(study, defaults, 'timeout') or 0.0)
        tmax = float(study_value(study, defaults, 'tmax'))
        print_step = int(study_value(study, defaults, 'printStep'))
        prandtl = float(study_value(study, defaults, 'prandtl'))
        gamma = float(study_value(study, defaults, 'gamma'))
        ny_ratio = int(study_value(study, defaults, 'ny_ratio') or 0)
        ny_fixed = int(study_value(study, defaults, 'ny') or 1)

        indices = range(0 if warmup else 1, repeats + 1)
        combos = itertools.product(backends, targets, nxs, plprs, viscosities, cores_list, indices)

        runs = []
        for backend, target, nx, plpr, viscosity, cores, repeat in combos:
            runs.append({
                'study': name,
                'kind': kind,
                'backend': str(backend),
                'target': str(target),
                'nx': int(nx),
                'ny': max(1, int(nx) // ny_ratio) if ny_ratio > 0 else ny_fixed,
                'plpr': float(plpr),
                'viscosity': float(viscosity),
                'cores': int(cores),
                'prandtl': prandtl,
                'gamma': gamma,
                'tmax': tmax,
                'printStep': print_step,
                'repeat': int(repeat),
                'warmup': repeat == 0,
                'timeout': timeout,
            })
        return runs
