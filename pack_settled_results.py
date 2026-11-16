"""Create a small review ZIP from one suite directory. CSV/NPZ stay on DICE."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite_directory', type=Path,
                        help='Directory printed after [SUITE] summary, without suite_summary.json')
    args=parser.parse_args()
    directory=args.suite_directory.resolve()
    summary=directory/'suite_summary.json'
    data=json.loads(summary.read_text())
    root=Path(__file__).resolve().parent
    files={str(p.relative_to(directory)):p for p in directory.glob('*')
           if p.is_file() and p.suffix in ('.json','.txt')}
    for result in data['results']:
        if 'run_report' in result:
            report=Path(result['run_report'])
            if report.is_file():
                files['run_reports/'+report.name]=report
                audit=report.with_name(report.stem+'_audit.json')
                if audit.is_file():
                    files['run_reports/'+audit.name]=audit
    # Only include source bytes when they still match this suite's fingerprint.
    # A later edit must never be presented as the source used by an older run.
    source_mismatches=[]
    for name,digest in data['source_sha256'].items():
        path=root/name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:
            if path.suffix in ('.py','.sh') or name=='expected_runtime_sha256.json':
                files['source/'+name]=path
        else:
            source_mismatches.append(name)
    output=directory.with_name(directory.name+'_review.zip')
    manifest=dict(scope=data['scope'],complete=data['complete'],passed=data['passed'],
                  passed_count=data.get('passed_count'),attempted_count=data.get('attempted_count'),
                  source_files_changed_since_suite=source_mismatches,
                  large_csv_and_npz_included=False,
                  files={name:hashlib.sha256(path.read_bytes()).hexdigest()
                         for name,path in files.items()})
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,path in sorted(files.items()):
            archive.write(path,name)
        archive.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        lines=['# Suite results', '', 'Scope: '+data['scope'], '',
               '| Case | Seed | Passed | Failure stage | Final 3D error mm | Final horizontal error mm | Final angle rad |',
               '| --- | --- | --- | --- | --- | --- | --- |']
        for result in data['results']:
            run=result.get('run',{})
            def metric(name,scale=1.):
                return f'{run[name]*scale:.6f}' if name in run else ''
            lines.append('| '+ ' | '.join([result['case'],str(result['seed']),
                str(result['passed']),result.get('failure_stage',''),
                metric('final_raw_goal_position_error_m',1000.),
                metric('final_horizontal_goal_error_m',1000.),
                metric('final_goal_orientation_error_rad')])+' |')
        lines += ['', 'A geometry-only pass is not a physical transport pass.',
                  'Large CSV and NPZ files remain at the paths in the run reports.',
                  'All final errors are against the original goal; no settling offset is subtracted.', '']
        archive.writestr('RESULTS.md','\n'.join(lines))
    print('Review ZIP:',output)
    print('Bytes:',output.stat().st_size)
    for result in data['results']:
        if not result['passed']:
            print('Failed case:',result['case'],'seed:',result['seed'],
                  'stage:',result.get('failure_stage','interrupted/incomplete'))
    if source_mismatches:
        print('Later changed source omitted:',', '.join(source_mismatches))
    print('Original CSV and NPZ remain on DICE.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
