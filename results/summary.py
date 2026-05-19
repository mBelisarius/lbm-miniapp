"""Writes the summary of the campaign in the three formats it is consumed in.

The JSON file keeps every aggregated number for later inspection, the Markdown
file is meant to be read while writing the text and the LaTeX file contains the
tables of the results chapter, already formatted with the Brazilian decimal
separator used in the thesis.
"""
import numpy as np

from common import write_json


def br_number(value, digits=3):
    return f"{value:.{digits}f}".replace('.', '{,}')


def br_scientific(value, digits=2, wrap=True):
    if value is None:
        return '---'
    exponent = int(np.floor(np.log10(abs(value)))) if value != 0 else 0
    mantissa = value / (10.0 ** exponent)
    body = f"{br_number(mantissa, digits)}\\times10^{{{exponent}}}"
    return f"\\({body}\\)" if wrap else body


def br_math(value, digits=3):
    return '---' if value is None else f"\\({br_number(value, digits)}\\)"


def br_times(value, digits=2):
    return '---' if value is None else f"\\({br_number(value, digits)}\\times\\)"


def strip_arrays(payload):
    """Removes the raw profile arrays before serialising the summary to JSON."""
    if isinstance(payload, dict):
        return {key: strip_arrays(value) for key, value in payload.items()
                if key not in ('simulation', 'analytical')}
    if isinstance(payload, list):
        return [strip_arrays(item) for item in payload]
    if isinstance(payload, np.generic):
        return payload.item()
    return payload


class SummaryWriter:
    """Serialises the aggregated campaign data."""

    def __init__(self, summary):
        self.summary = summary

    def json(self, path, figures=None):
        payload = strip_arrays(self.summary)
        payload['figures'] = {name: (str(figure) if figure else None)
                              for name, figure in (figures or {}).items()}
        write_json(path, payload)
        print(f"Summary written to {path}")
        return path

    def markdown(self, path):
        lines = ['# Campaign summary for the results chapter', '']
        lines += self._markdown_convergence()
        lines += self._markdown_pressure_ratio()
        lines += self._markdown_viscosity()
        lines += self._markdown_scalability()
        lines += self._markdown_technologies()
        lines += self._markdown_complexity()
        lines += self._markdown_profiles()

        path.write_text('\n'.join(lines) + '\n')
        print(f"Summary written to {path}")
        return path

    def _markdown_convergence(self):
        convergence = self.summary.get('convergence')
        if not convergence or not convergence['rows']:
            return []

        lines = [f"## Convergence (pressure ratio {convergence['pressure_ratio']:g})", '',
                 '| nx | RMSE(rho) | order | RMSE(u) | order | RMSE(p) | order |',
                 '|---|---|---|---|---|---|---|']
        for row in convergence['rows']:
            cells = [str(row['nx'])]
            for variable in ('rho', 'u', 'p'):
                cells.append(f"{row['rmse'][variable]:.3e}")
                value = row['order'].get(variable)
                cells.append(f"{value:.3f}" if value else '---')
            lines.append('| ' + ' | '.join(cells) + ' |')

        if convergence['order_range']:
            lines += ['', f"Effective order range: {convergence['order_range'][0]:.3f} "
                          f"to {convergence['order_range'][1]:.3f}"]
        if convergence['error_reduction']:
            lines += [f"Pressure error reduced {convergence['error_reduction']:.1f} times for a "
                      f"{convergence['refinement']:.0f} times refinement"]
        return lines + ['']

    def _markdown_pressure_ratio(self):
        ratios = self.summary.get('pressure_ratio')
        if not ratios or not ratios['rmse_pressure']:
            return []

        header = ' | '.join(f"nx={nx}" for nx in ratios['nx_values'])
        lines = ['## Effect of the pressure ratio on the error', '',
                 f"| Pi | {header} | mean order |",
                 '|---' * (len(ratios['nx_values']) + 2) + '|']
        for ratio in sorted(ratios['rmse_pressure']):
            cells = []
            for nx in ratios['nx_values']:
                value = ratios['rmse_pressure'][ratio].get(nx)
                cells.append(f"{value:.3e}" if value else 'diverged')
            order = ratios['mean_order'].get(ratio)
            lines.append(f"| {ratio:g} | " + ' | '.join(cells) + ' | '
                         + (f"{order:.3f} |" if order else '--- |'))

        if ratios['stability']['limit']:
            lines += ['', f"Highest fully stable pressure ratio: {ratios['stability']['limit']:g}"]
        for diverged in ratios['stability']['diverged']:
            fraction = diverged.get('time_fraction')
            lines.append(f"Diverged: Pi={diverged['plpr']:g}, nx={diverged['nx']}"
                         + (f", reached {100 * fraction:.0f}% of tmax" if fraction else ''))
        return lines + ['']

    def _markdown_viscosity(self):
        viscosity = self.summary.get('viscosity')
        if not viscosity or not viscosity['rows']:
            return []

        lines = ['## Effect of the viscosity on the error', '',
                 '| nu | RMSE(p) | deviation from the minimum |', '|---|---|---|']
        for row in viscosity['rows']:
            deviation = row.get('deviation')
            lines.append(f"| {row['viscosity']:.3e} | {row['rmse_p']:.3e} | "
                         + (f"{100 * deviation:+.0f}% |" if deviation else '--- |'))

        if viscosity['optimum_viscosity']:
            lines += ['', f"Minimum error at nu = {viscosity['optimum_viscosity']:.3e}"]
        if viscosity['stability_limit']:
            lines += [f"Simulation diverges at nu <= {viscosity['stability_limit']:.3e}"]
        return lines + ['']

    def _markdown_scalability(self):
        scalability = self.summary.get('cpu_scalability')
        if not scalability or not scalability['per_nx']:
            return []

        lines = ['## CPU scalability', '',
                 '| nx | cores | runtime (s) | speedup | efficiency | serial fraction | MLUPS |',
                 '|---|---|---|---|---|---|---|']
        for nx, rows in sorted(scalability['per_nx'].items()):
            for row in rows:
                lines.append(
                    f"| {nx} | {row['cores']} | {row['runtime']:.6f} | "
                    + (f"{row['speedup']:.3f} | " if row['speedup'] is not None else '--- | ')
                    + (f"{row['efficiency']:.3f} | " if row['efficiency'] is not None else '--- | ')
                    + (f"{row['serial_fraction']:.3f} | " if row['serial_fraction'] is not None else '--- | ')
                    + (f"{row['mlups']:.1f} |" if row['mlups'] is not None else '--- |'))
        return lines + ['']

    def _markdown_technologies(self):
        lines = []
        for key, title in (('cpu_technologies', 'CPU technologies'),
                           ('gpu_technologies', 'GPU technologies')):
            technologies = self.summary.get(key)
            if not technologies or not technologies['summary']:
                continue
            lines += [f"## {title} (nx = {technologies['largest_nx']})", '',
                      '| technology | MLUPS | speedup |', '|---|---|---|']
            for backend, values in sorted(technologies['summary'].items()):
                lines.append(f"| {backend} | "
                             + (f"{values['mlups']:.1f} | " if values['mlups'] else '--- | ')
                             + (f"{values['speedup']:.2f} |" if values['speedup'] else '--- |'))
            crossings = {b: v for b, v in technologies['break_even'].items() if v}
            for backend, crossing in sorted(crossings.items()):
                lines.append(f"Break-even of {backend} at nx ~ {crossing:.0f}")
            lines.append('')
        return lines

    def _markdown_complexity(self):
        complexity = self.summary.get('complexity')
        if not complexity or not complexity['technologies']:
            return []

        lines = ['## Code complexity', '', '| technology | LOC | relative |', '|---|---|---|']
        for technology, values in sorted(complexity['technologies'].items()):
            lines.append(f"| {technology} | {values['loc']} | "
                         + (f"{values['relative']:.2f} |" if values['relative'] else '--- |'))
        return lines + ['']

    def _markdown_profiles(self):
        profiles = self.summary.get('profiles')
        if not profiles:
            return []

        lines = ['## Wave front thickness of the reference run', '',
                 '| t (s) | shock (nodes) | contact (nodes) |', '|---|---|---|']
        for panel in profiles['panels']:
            thickness = panel.get('thickness') or {}
            shock, contact = thickness.get('shock_nodes'), thickness.get('contact_nodes')
            lines.append(f"| {panel['time']:.3f} | {'---' if shock is None else shock} | "
                         f"{'---' if contact is None else contact} |")

        if profiles['contact_growth_ratio']:
            ratios = ', '.join(f"{r:.2f}" for r in profiles['contact_growth_ratio'])
            lines += ['', f"Contact thickness growth ratio between consecutive instants: {ratios}"]
        return lines + ['']

    def latex(self, path):
        """Writes the tables of the results chapter with Brazilian number format."""
        blocks = [block for block in (self._latex_convergence(),
                                      self._latex_pressure_ratio(),
                                      self._latex_viscosity(),
                                      self._latex_scalability(),
                                      *self._latex_technologies(),
                                      self._latex_complexity()) if block]

        header = ('% Tables generated by lbmini/Analysis/campaign.py.\n'
                  '% Numbers follow the Brazilian decimal separator used in the thesis.\n')
        path.write_text(header + '\n\n'.join(blocks) + '\n')
        print(f"LaTeX tables written to {path}")
        return path

    @staticmethod
    def _table(caption, label, alignment, header, rows, footer=None):
        """Assembles a booktabs table in the layout used by the thesis."""
        lines = [
            r'\begin{table}[H]',
            r'    \centering',
            f"    \\caption{{{caption}}}",
            f"    \\label{{{label}}}",
            f"    \\begin{{tabular}}{{{alignment}}}",
            r'        \toprule',
            '        ' + header + r' \\',
            r'        \midrule',
            '\n'.join('        ' + row + r' \\' for row in rows),
        ]
        if footer:
            lines += [r'        \midrule', '        ' + footer + r' \\']
        lines += [
            r'        \bottomrule',
            r'    \end{tabular}',
            r'    \fonte{O autor (2026).}',
            r'\end{table}',
        ]
        return '\n'.join(lines)

    def _latex_convergence(self):
        convergence = self.summary.get('convergence')
        if not convergence or not convergence['rows']:
            return None

        rows = []
        for row in convergence['rows']:
            order = row['order']
            rows.append(' & '.join([
                str(row['nx']),
                br_scientific(row['rmse']['rho']),
                br_math(order.get('rho')),
                br_scientific(row['rmse']['u']),
                br_math(order.get('u')),
                br_scientific(row['rmse']['p']),
                br_math(order.get('p')),
            ]))

        return self._table(
            'Erro quadrático médio (EQM) e ordem de convergência efetiva',
            'tab:convergencia-eqm', 'ccccccc',
            r'\(N_x\) & \(\mathrm{EQM}(\rho)\) & \(p_{ef}\) & \(\mathrm{EQM}(u)\)'
            r' & \(p_{ef}\) & \(\mathrm{EQM}(p)\) & \(p_{ef}\)',
            rows)

    def _latex_pressure_ratio(self):
        ratios = self.summary.get('pressure_ratio')
        if not ratios or not ratios['rmse_pressure']:
            return None

        nx_values = ratios['nx_values']
        rows = []
        for ratio in sorted(ratios['rmse_pressure']):
            cells = [f"{ratio:g}"]
            for nx in nx_values:
                value = ratios['rmse_pressure'][ratio].get(nx)
                cells.append(br_scientific(value) if value else 'divergente')
            rows.append(' & '.join(cells))

        orders = '; '.join(
            f"{br_number(order, 3)} (\\(\\Pi={ratio:g}\\))"
            for ratio, order in sorted(ratios['mean_order'].items()) if order)
        footer = f"\\multicolumn{{{len(nx_values) + 1}}}{{l}}{{Ordem efetiva média: {orders}}}"

        return self._table(
            'Erro quadrático médio da pressão por razão de pressões e número de nós',
            'tab:erro-razao-pressoes', 'c' + 'c' * len(nx_values),
            r'\(\Pi\) & ' + ' & '.join(f"\\(N_x={nx}\\)" for nx in nx_values),
            rows, footer)

    def _latex_viscosity(self):
        viscosity = self.summary.get('viscosity')
        if not viscosity or not viscosity['rows']:
            return None

        rows = []
        for row in viscosity['rows']:
            deviation = row.get('deviation')
            rows.append(' & '.join([
                br_scientific(row['viscosity']),
                br_scientific(row['rmse_p']),
                (f"\\({int(round(100 * deviation)):+d}\\%\\)" if deviation else '---'),
            ]))

        limit = viscosity['stability_limit']
        footer = (f"\\multicolumn{{3}}{{l}}{{\\(\\nu \\lesssim "
                  f"{br_scientific(limit, wrap=False)}\\): "
                  f"simulação divergente antes de \\(t_{{max}}\\)}}") if limit else None

        return self._table(
            'Erro quadrático médio da pressão por viscosidade cinemática',
            'tab:erro-viscosidade', 'ccc',
            r'\(\nu\) & \(\mathrm{EQM}(p)\) & Desvio ao mínimo',
            rows, footer)

    def _latex_scalability(self):
        scalability = self.summary.get('cpu_scalability')
        if not scalability or not scalability['per_nx']:
            return None

        rows = []
        for nx, per_cores in sorted(scalability['per_nx'].items()):
            top = per_cores[-1]
            rows.append(' & '.join([
                str(nx),
                br_math(top['speedup'], 2),
                br_math(top['efficiency'], 2),
                br_math(top['serial_fraction'], 3),
                br_math(top['mlups'], 1),
            ]))

        return self._table(
            'Aceleração, eficiência paralela, fração sequencial efetiva e taxa de'
            ' atualização de nós no maior número de núcleos ensaiado',
            'tab:escalabilidade-cpu', 'ccccc',
            r'\(N_x\) & \(S\) & \(\eta\) & \(f_{ef}\) & Taxa de atualização (MLUPS)',
            rows)

    def _latex_technologies(self):
        blocks = []
        for key, label, caption in (
            ('cpu_technologies', 'tab:desempenho-cpu-tecnologias',
             'Resumo de desempenho das tecnologias de paralelização em CPU'),
            ('gpu_technologies', 'tab:desempenho-gpu-tecnologias',
             'Resumo de desempenho das tecnologias de paralelização em GPU'),
        ):
            technologies = self.summary.get(key)
            if not technologies or not technologies['summary']:
                continue

            rows = [' & '.join([backend, br_math(values['mlups'], 1), br_times(values['speedup'])])
                    for backend, values in sorted(technologies['summary'].items())]
            blocks.append(self._table(
                f"{caption} para malha de \\(N_x={technologies['largest_nx']}\\) nós",
                label, 'lcc',
                r'Tecnologia & Taxa de atualização (MLUPS) & Aceleração (\(S\))',
                rows))
        return blocks

    def _latex_complexity(self):
        complexity = self.summary.get('complexity')
        if not complexity or not complexity['technologies']:
            return None

        rows = [' & '.join([technology, str(values['loc']), br_times(values['relative'])])
                for technology, values in sorted(complexity['technologies'].items())]

        return self._table(
            'Complexidade relativa de implementação por tecnologia',
            'tab:complexidade-loc-tecnologias', 'lcc',
            r'Tecnologia & LOC (absoluto) & LOC relativo à versão sequencial',
            rows)
