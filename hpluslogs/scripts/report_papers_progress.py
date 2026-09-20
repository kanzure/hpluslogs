#!/usr/bin/env python3
"""Persist periodic remote conversion reports until the container exits."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def capture(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=300)
    args = parser.parse_args()
    if args.interval < 10:
        parser.error('--interval must be at least 10 seconds.')
    while True:
        state = json.loads(capture(['docker', 'inspect', '--format', '{{json .State}}', args.container]))
        if state['Running']:
            command = ['docker', 'exec', args.container, 'python', '-m', 'hpluslogs.conversion_cli', 'progress']
        else:
            image = capture(['docker', 'inspect', '--format', '{{.Config.Image}}', args.container]).strip()
            user = capture(['docker', 'inspect', '--format', '{{.Config.User}}', args.container]).strip()
            command = ['docker', 'run', '--rm', '--network', 'none', '--user', user,
                       '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                       '--tmpfs', '/tmp:rw,size=64m', '--mount', f'type=bind,src={args.data_dir},dst=/data',
                       image, 'progress']
        report = json.loads(capture(command))
        report['container_state'] = {key: state.get(key) for key in ('Status', 'Running', 'ExitCode', 'OOMKilled')}
        target = args.data_dir/'papers2_progress_latest.json'
        temp = target.with_suffix('.tmp'); temp.write_text(json.dumps(report, indent=2)+'\n'); temp.replace(target)
        with (args.data_dir/'papers2_progress_history.jsonl').open('a') as stream:
            stream.write(json.dumps(report)+'\n')
        summary = {key: report[key] for key in ('measured_at', 'ready', 'pdf_count', 'failed', 'markdown_mib')}
        summary['container_state'] = report['container_state']
        summary['eta_hours'] = report['throughput'][0]['first_pass_remaining_hours'] if state['Running'] else None
        print(json.dumps(summary), flush=True)
        if not state['Running']:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
