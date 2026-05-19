"""Executes the runs of a campaign, one at a time and resumably.

Every run gets its own directory with the configuration, the log and the output
of the executable. The outcome is appended to the ledger as soon as it is known,
so an interruption only discards the run in progress.
"""
import os
import shutil
import subprocess

from pathlib import Path

from common import (CampaignInterrupt, REPO_ROOT, dump_yaml, load_yaml, resolve_path,
                    write_json)
from ledger import Ledger
from plan import CampaignPlan, build_run_config, run_id, run_key
from simulation import measure_run


class CampaignRunner:
    """Runs the pending combinations of a campaign."""

    def __init__(self, campaign, overrides=None):
        self.campaign = campaign
        self.workdir = resolve_path(campaign['workdir'])
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.base_config = load_yaml(resolve_path(campaign['base_config']))
        self.ledger = Ledger(self.workdir / 'ledger.json')
        self.plan = CampaignPlan(campaign, overrides)

    def runs(self, studies=None):
        return self.plan.runs(studies)

    def pending(self, runs, retry_failed=False):
        """Selects the runs that were never finished, or that must be redone."""
        pending = []
        for run in runs:
            entry = self.ledger.get(run_key(run))
            if entry is None or (retry_failed and entry.get('status') != 'completed'):
                pending.append(run)
        return pending

    def describe(self, runs):
        """Prints the planned runs and the state recorded for each of them."""
        print(f"{len(runs)} runs planned:")
        for run in runs:
            entry = self.ledger.get(run_key(run))
            status = entry.get('status') if entry else 'pending'
            print(f"  [{status:>9}] {run_key(run)}")

    def execute(self, runs, retry_failed=False, keep_data='all'):
        """Executes every pending run; returns False if it was interrupted."""
        pending = self.pending(runs, retry_failed)
        print(f"{len(runs)} runs in the campaign, {len(pending)} to execute.")

        for index, run in enumerate(pending, start=1):
            key = run_key(run)
            run_dir = self.workdir / 'runs' / run['study'] / run_id(run)
            print(f"[{index}/{len(pending)}] {key}")

            try:
                status, detail = self._simulate(run, run_dir)
            except (CampaignInterrupt, KeyboardInterrupt):
                print("\nInterrupted: discarding the run in progress.")
                shutil.rmtree(run_dir, ignore_errors=True)
                self.ledger.drop(key)
                print("Campaign stopped. Run the script again to continue from this point.")
                return False

            entry = self._record(run, run_dir, status, detail)
            self.ledger.set(key, entry)
            self._report(entry)

            if keep_data == 'none' and entry['status'] != 'completed':
                shutil.rmtree(run_dir, ignore_errors=True)

        print("All planned runs are finished.")
        return True

    def _record(self, run, run_dir, status, detail):
        """Builds the ledger entry of a finished run, measuring its output."""
        entry = {'params': run, 'status': status,
                 'directory': str(run_dir.relative_to(REPO_ROOT))}

        if status == 'completed':
            try:
                entry['metrics'] = measure_run(run, run_dir)
            except Exception as e:
                entry['status'] = 'failed'
                entry['error'] = f"measurement failed: {e}"
            else:
                if entry['metrics'].get('diverged'):
                    entry['status'] = 'diverged'
            return entry

        entry['error'] = f"return code {detail}" if detail is not None else 'timeout'
        # A diverging simulation still produces partial output worth keeping.
        try:
            entry['metrics'] = measure_run(run, run_dir)
            entry['status'] = 'diverged'
        except Exception:
            pass
        return entry

    @staticmethod
    def _report(entry):
        metrics = entry.get('metrics', {})
        print(f"  status: {entry['status']}"
              + (f", runtime {metrics['runtime']:.3f} s" if metrics.get('runtime') else '')
              + (f", RMSE(p) {metrics['rmse']['p']:.3e}" if metrics.get('rmse') else ''))

    def _simulate(self, run, run_dir):
        """Executes a single simulation, discarding the directory if interrupted."""
        run_dir = Path(run_dir)
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True)

        config_path = run_dir / 'config.yaml'
        dump_yaml(config_path, build_run_config(self.base_config, run))
        write_json(run_dir / 'run.json', run)

        command = self._launcher(run) + [
            str(self._executable(run)),
            '--config', str(config_path),
            '--outpath', str(run_dir / 'out'),
        ]
        print(f"  command: {' '.join(command)}")

        timeout = run['timeout'] if run['timeout'] > 0 else None
        with open(run_dir / 'run.log', 'w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=self._environment(run), cwd=str(REPO_ROOT))
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return 'timeout', None
            except (CampaignInterrupt, KeyboardInterrupt):
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise

        return ('completed' if return_code == 0 else 'failed'), return_code

    def _executable(self, run):
        override = (self.campaign.get('executables', {}) or {}).get(run['backend'])
        return resolve_path(override or self.campaign['executable'])

    def _launcher(self, run):
        template = (self.campaign.get('launchers', {}) or {}).get(run['backend'])
        if not template:
            return []
        cores = run['cores'] if run['cores'] > 0 else 1
        return [str(item).format(cores=cores) for item in template]

    def _environment(self, run):
        env = os.environ.copy()
        # Settings declared by the campaign, such as the thread affinity,
        # override the inherited ones (container images ship their own).
        env.update({str(name): str(value)
                    for name, value in (self.campaign.get('environment', {}) or {}).items()})
        if run['cores'] > 0:
            env['OMP_NUM_THREADS'] = str(run['cores'])
            env['ACC_NUM_CORES'] = str(run['cores'])
        return env
