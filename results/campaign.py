"""Command line of the analysis tools of the mini-application.

The campaign is described by a YAML file (see campaign.yaml). Each study is
expanded into the cartesian product of its parameters and every combination is
executed as an independent simulation inside its own directory.

The runner is resumable: a ledger file records the outcome of every finished
combination, so an interrupted campaign restarts exactly where it stopped. The
combination that was running when the interruption happened is discarded, its
directory removed, and it is executed again on the next invocation.

The report stage consumes the ledger, derives every number required by the
results chapter (RMSE, effective convergence order, speedup, parallel
efficiency, effective serial fraction, node update rate and code complexity),
writes the summary files and generates the figures.

The remaining commands analyse an arbitrary tree of simulation outputs, without
any campaign: the plots and the report of a single run, the error against the
analytical solution, the speedup per number of cores and the animation of one
run. They are meant for debugging a run before or outside a campaign.

Usage:
    python campaign.py run                             # execute the pending runs
    python campaign.py report                          # summary and figures
    python campaign.py                                 # run followed by report
    python campaign.py single --outpath DIR            # plots and report of one run
    python campaign.py errors --outpath DIR            # error of every out* directory
    python campaign.py scalability --outpath DIR       # speedup per number of cores
    python campaign.py movie --outpath DIR             # animation of one run

The implementation is split by responsibility: common.py (paths and files),
plan.py (expansion of the studies), ledger.py (persistent state), runner.py
(execution), simulation.py (reading and measuring one run), aggregate.py
(numbers of the chapter), figures.py (plots), summary.py (JSON, Markdown and
LaTeX), report.py (orchestration) and single.py, outputs.py plus movie.py
(standalone analyses).
"""
import argparse
import sys

from common import DEFAULT_CAMPAIGN, install_signal_handlers, load_yaml, resolve_path, select_backend


COMMANDS = ('run', 'report', 'all', 'single', 'errors', 'scalability', 'movie')


def parse_arguments(argv):
    # Without a command the campaign is run and reported, as in 'all'.
    if not argv or argv[0] not in COMMANDS:
        argv = ['all'] + list(argv)

    parser = argparse.ArgumentParser(
        description='Run the measurement campaign of the results chapter and report it.')
    commands = parser.add_subparsers(dest='command')

    campaign = argparse.ArgumentParser(add_help=False)
    campaign.add_argument('--campaign', type=str, default=str(DEFAULT_CAMPAIGN),
                          help='Path to the campaign YAML file.')
    campaign.add_argument('--exepath', type=str, default=None,
                          help='Overrides the executable declared in the campaign file.')
    campaign.add_argument('--workdir', type=str, default=None,
                          help='Overrides the working directory declared in the campaign file.')
    campaign.add_argument('--figpath', type=str, default=None,
                          help='Overrides the figure directory declared in the campaign file.')
    campaign.add_argument('--lang', type=str, choices=['en', 'br'], default=None,
                          help='Language for figure texts: en (English), br (Português).')

    execution = argparse.ArgumentParser(add_help=False)
    execution.add_argument('--studies', nargs='+', default=None,
                           help='Subset of studies to execute.')
    execution.add_argument('--nx', nargs='+', type=int, default=None,
                           help='Overrides the lattice numbers of every study.')
    execution.add_argument('--plpr', nargs='+', type=float, default=None,
                           help='Overrides the pressure ratios of every study.')
    execution.add_argument('--dry-run', action='store_true',
                           help='List the planned runs and their state without executing them.')
    execution.add_argument('--retry-failed', action='store_true',
                           help='Execute again the runs that failed or diverged.')
    execution.add_argument('--keep-data', type=str, choices=['all', 'none'], default='all',
                           help='Keep the output of failed runs (all) or delete it (none).')

    reporting = argparse.ArgumentParser(add_help=False)
    reporting.add_argument('--reanalyse', action='store_true',
                           help='Recompute the metrics of every stored run before reporting.')

    commands.add_parser('run', parents=[campaign, execution],
                        help='Execute the pending runs of the campaign.')
    commands.add_parser('report', parents=[campaign, reporting],
                        help='Aggregate the ledger, write the summary and the figures.')
    commands.add_parser('all', parents=[campaign, execution, reporting],
                        help='Execute the pending runs and then report them.')

    single = commands.add_parser('single', help='Plots and report of a single simulation output.')
    single.add_argument('--outpath', type=str, default='out',
                        help='Path to the simulation output directory.')
    single.add_argument('--figpath', type=str, default=None,
                        help='Directory of the figures and of the report. '
                             'Defaults to the output directory itself.')
    single.add_argument('--times', nargs='+', type=float, default=None,
                        help='Instants, in seconds, whose profiles are plotted.')
    single.add_argument('--panels', type=int, default=3,
                        help='Number of profile panels when no instant is given.')
    single.add_argument('--lang', type=str, choices=['en', 'br'], default='en',
                        help='Language for figure texts: en (English), br (Português).')

    errors = commands.add_parser('errors', help='Error of every out* directory of a tree.')
    errors.add_argument('--outpath', type=str, default='.',
                        help='Directory containing the out* subdirectories.')
    errors.add_argument('--variable', type=str, choices=['rho', 'u', 'p'], default='rho',
                        help='Variable whose error is plotted.')
    errors.add_argument('--lang', type=str, choices=['en', 'br'], default='en',
                        help='Language for figure texts: en (English), br (Português).')

    scalability = commands.add_parser('scalability', help='Speedup per number of cores.')
    scalability.add_argument('--outpath', type=str, default='simulations-scalability',
                             help='Directory containing the out* subdirectories.')
    scalability.add_argument('--maxcores', type=int, default=None,
                             help='Maximum number of cores to include and plot.')
    scalability.add_argument('--lang', type=str, choices=['en', 'br'], default='en',
                             help='Language for figure texts: en (English), br (Português).')

    movie = commands.add_parser('movie', help='Animation of a single simulation output.')
    movie.add_argument('--outpath', type=str, default='out',
                       help='Path to the simulation output directory.')
    movie.add_argument('--config', type=str, default=None,
                       help='Path to the config.yaml file of the run.')
    movie.add_argument('--speedup', type=float, default=1.0,
                       help='Playback speedup factor. >1 for faster, <1 for slower.')
    movie.add_argument('--show', action='store_true',
                       help='Display the animation besides saving it.')

    return parser.parse_args(argv)


def load_campaign(args):
    """Loads the campaign file and applies the overrides of the command line."""
    campaign = load_yaml(args.campaign)
    if args.exepath:
        campaign['executable'] = args.exepath
    if args.workdir:
        campaign['workdir'] = args.workdir
    if args.figpath:
        campaign.setdefault('figures', {})['path'] = args.figpath
    if args.lang:
        campaign.setdefault('figures', {})['lang'] = args.lang
    return campaign


def stage_run(args):
    """Executes the pending runs of the campaign."""
    from runner import CampaignRunner

    runner = CampaignRunner(load_campaign(args), {'nx': args.nx, 'plpr': args.plpr})
    runs = runner.runs(args.studies)

    if args.dry_run:
        runner.describe(runs)
        return True

    return runner.execute(runs, retry_failed=args.retry_failed, keep_data=args.keep_data)


def stage_report(args):
    """Aggregates the ledger and writes the summary and the figures."""
    from report import ReportBuilder

    return ReportBuilder(load_campaign(args)).build(reanalyse=args.reanalyse)


def stage_single(args):
    """Writes the plots and the report of a single simulation output."""
    from figures import FigureMaker
    from single import SingleRunReport

    run_dir = resolve_path(args.outpath)
    figures = FigureMaker(resolve_path(args.figpath) if args.figpath else run_dir, lang=args.lang)
    report = SingleRunReport(run_dir, figures, times=args.times, panels=args.panels)
    return bool(report.build())


def stage_errors(args):
    """Plots the error of every output directory of a tree."""
    from figures import FigureMaker
    from outputs import plot_errors

    base_dir = resolve_path(args.outpath)
    return plot_errors(base_dir, FigureMaker(base_dir, lang=args.lang),
                       variable=args.variable) is not None


def stage_scalability(args):
    """Plots the speedup of every output directory of a tree."""
    from figures import FigureMaker
    from outputs import plot_scalability

    base_dir = resolve_path(args.outpath)
    return plot_scalability(base_dir, FigureMaker(base_dir, lang=args.lang),
                            maxcores=args.maxcores) is not None


def stage_movie(args):
    """Renders the animation of a single simulation output."""
    from movie import SimulationMovie

    movie = SimulationMovie(resolve_path(args.outpath), config=args.config, speedup=args.speedup)
    return movie.render(show=args.show) is not None


def main(argv=None):
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    select_backend(interactive=getattr(args, 'show', False))
    install_signal_handlers()

    if args.command == 'run':
        return 0 if stage_run(args) else 1
    if args.command == 'report':
        return 0 if stage_report(args) else 1
    if args.command == 'all':
        if not stage_run(args):
            return 1
        if args.dry_run:
            return 0
        return 0 if stage_report(args) else 1
    if args.command == 'single':
        return 0 if stage_single(args) else 1
    if args.command == 'errors':
        return 0 if stage_errors(args) else 1
    if args.command == 'scalability':
        return 0 if stage_scalability(args) else 1
    if args.command == 'movie':
        return 0 if stage_movie(args) else 1

    return 1


if __name__ == '__main__':
    sys.exit(main())
