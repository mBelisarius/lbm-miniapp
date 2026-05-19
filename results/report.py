"""Turns the ledger of a campaign into the summary and the figures.

The builder aggregates the recorded runs, draws the figures of the results
chapter and writes the summary in JSON, Markdown and LaTeX.
"""
import aggregate

from common import REPO_ROOT, resolve_path
from figures import FigureMaker
from ledger import Ledger
from simulation import measure_run
from summary import SummaryWriter


class ReportBuilder:
    """Builds every artefact of the results chapter from a finished campaign."""

    def __init__(self, campaign):
        self.campaign = campaign
        self.workdir = resolve_path(campaign['workdir'])
        self.ledger = Ledger(self.workdir / 'ledger.json')
        self.sections = campaign.get('report', {}) or {}

        figures = campaign.get('figures', {}) or {}
        self.figures = FigureMaker(resolve_path(figures.get('path', 'thesis/fig')),
                                   lang=figures.get('lang', 'en'),
                                   dpi=figures.get('dpi', 200))

    def build(self, reanalyse=False):
        """Aggregates, plots and writes; returns False when there is no data."""
        if not self.ledger.entries:
            print(f"No results found in {self.workdir / 'ledger.json'}; run the campaign first.")
            return False

        if reanalyse:
            self.reanalyse()

        summary = self.aggregate()
        figures = self.plot(summary)

        writer = SummaryWriter(summary)
        writer.json(self.workdir / 'summary.json', figures)
        writer.markdown(self.workdir / 'summary.md')
        writer.latex(self.workdir / 'summary.tex')
        return True

    def reanalyse(self):
        """Recomputes the metrics of every stored run before aggregating."""
        for key, entry in list(self.ledger.entries.items()):
            if entry.get('status') not in ('completed', 'diverged'):
                continue
            try:
                entry['metrics'] = measure_run(entry['params'], REPO_ROOT / entry['directory'])
            except Exception as e:
                print(f"Warning: could not re-analyse {key}: {e}")
        self.ledger.save()

    def aggregate(self):
        """Derives every number required by the results chapter."""
        sections = self.sections
        summary = {
            'convergence': aggregate.convergence(self.ledger, sections.get('convergence', {})),
            'pressure_ratio': aggregate.pressure_ratio(self.ledger, sections.get('pressure_ratio', {})),
            'viscosity': aggregate.viscosity(self.ledger, sections.get('viscosity', {})),
            'cpu_scalability': aggregate.scalability(self.ledger, sections.get('cpu_scalability', {})),
            'complexity': aggregate.complexity(self.campaign),
            'profiles': aggregate.profiles(self.ledger, sections.get('profiles', {})),
        }

        cpu_section = sections.get('cpu_technologies')
        if cpu_section:
            summary['cpu_technologies'] = aggregate.technologies(self.ledger, cpu_section)

        gpu_section = sections.get('gpu_technologies')
        if gpu_section:
            baseline_study = gpu_section.get('baseline_study')
            baseline = self.ledger.completed(baseline_study) if baseline_study else None
            summary['gpu_technologies'] = aggregate.technologies(self.ledger, gpu_section, baseline)

        return summary

    def plot(self, summary):
        """Draws every figure of the chapter and returns their paths."""
        figures = {
            'profiles': self.figures.profiles(summary['profiles']),
            'convergence': self.figures.convergence(summary['pressure_ratio']),
            'pressure_ratio': self.figures.pressure_ratio(summary['pressure_ratio']),
            'viscosity': self.figures.viscosity(summary['viscosity']),
            'scalability': self.figures.scalability(summary['cpu_scalability']),
        }

        if 'cpu_technologies' in summary:
            figures['cpu_technologies'] = self.figures.technologies(
                summary['cpu_technologies'], 'cpu-technologies.png')
        if 'gpu_technologies' in summary:
            figures['gpu_technologies'] = self.figures.technologies(
                summary['gpu_technologies'], 'gpu-technologies.png', break_even_line=True)

        return figures
