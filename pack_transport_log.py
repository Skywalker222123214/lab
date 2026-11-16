"""Pack full-rate transport observations and full-column statistics for review.

Python standard library only. The original CSV is read without modification.
This is a log transfer utility, not a controller or coursework runtime module.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile


def pack(source, output=None):
    source = Path(source).resolve()
    output = (Path(output).resolve() if output else
              source.with_name(source.stem + '_review.zip'))
    if source == output:
        raise ValueError('Output must differ from original CSV')
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    summary = dict(source_name=source.name, source_bytes=source.stat().st_size,
                   source_sha256=digest.hexdigest(), rows=0,
                   nonfinite_cells=0, unexpected_time_steps=0,
                   expected_time_step_s=0.001,
                   sampling='Every original row; no downsampling or rounding',
                   limitation='Joint/torque time series remain in original CSV; '
                              'their full-column extrema are included here.')
    phases = Counter()
    previous = None
    with source.open(newline='') as stream, zipfile.ZipFile(
            output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        required = {'time_s', 'phase', 'step_allowed', 'obstacle_load_N',
                    'cube_x_m', 'cube_y_m', 'cube_z_m', 'torque_clipped'}
        if not fields or not required.issubset(fields):
            raise ValueError('Not a recognized transport CSV')
        retained = [name for name in fields if not name.startswith(
            ('q_', 'v_', 'tau_next_', 'tau_applied_previous_'))]
        stats = {name: dict(min=None, max=None, nonfinite=0)
                 for name in fields if name != 'phase'}
        with archive.open('transport_metrics.csv', 'w') as raw:
            with io.TextIOWrapper(raw, encoding='utf-8', newline='') as compact:
                writer = csv.DictWriter(compact, fieldnames=retained)
                writer.writeheader()
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError('Malformed CSV row ' + str(summary['rows'] + 2))
                    values = {name: float(row[name]) for name in stats}
                    for name, value in values.items():
                        entry = stats[name]
                        if not math.isfinite(value):
                            entry['nonfinite'] += 1
                            summary['nonfinite_cells'] += 1
                        else:
                            if entry['min'] is None or value < entry['min']:
                                entry['min'] = value
                            if entry['max'] is None or value > entry['max']:
                                entry['max'] = value
                    time_s = values['time_s']
                    if previous is not None and abs(time_s - previous - .001) > 1e-8:
                        summary['unexpected_time_steps'] += 1
                    previous = time_s
                    compact_row = {name: row[name] for name in retained}
                    if summary['rows'] == 0:
                        summary['first_row'] = compact_row
                    summary['last_row'] = compact_row
                    summary['rows'] += 1
                    phases[row['phase']] += 1
                    writer.writerow(compact_row)
        summary['phases'] = dict(phases)
        summary['all_numeric_columns'] = stats
        summary['retained_columns'] = retained
        archive.writestr('summary.json', json.dumps(summary, indent=2) + '\n')
    print('Review ZIP:', output)
    print('Size: %.2f MiB; original: %.2f MiB; rows: %d' % (
        output.stat().st_size / 2**20, source.stat().st_size / 2**20, summary['rows']))
    print('Original CSV preserved. Send the review ZIP; keep the original locally.')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_file', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    pack(args.csv_file, args.output)
