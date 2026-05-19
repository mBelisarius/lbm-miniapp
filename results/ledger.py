"""Persistent record of the runs already finished by the campaign.

The ledger is the only state kept between invocations: it makes the campaign
resumable and feeds the report stage.
"""
import json

from pathlib import Path

from common import write_json


class Ledger:
    """Persistent record of the runs already finished."""

    def __init__(self, path):
        self.path = Path(path)
        self.entries = {}
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    self.entries = json.load(f).get('runs', {})
            except (OSError, ValueError) as e:
                print(f"Warning: could not read ledger {self.path}: {e}")

    def get(self, key):
        return self.entries.get(key)

    def set(self, key, entry):
        self.entries[key] = entry
        self.save()

    def drop(self, key):
        if key in self.entries:
            del self.entries[key]
            self.save()

    def save(self):
        write_json(self.path, {'runs': self.entries})

    def by_study(self, study):
        return [entry for entry in self.entries.values()
                if entry.get('params', {}).get('study') == study]

    def completed(self, study):
        return [entry for entry in self.by_study(study) if entry['status'] == 'completed']
