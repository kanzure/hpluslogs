"""Run one independent OCR repair pass after the current converter exits."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def inspect(name):
    return json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1:
        parser.error('Workers and timeout must be positive.')
    original = inspect(args.container)
    name = args.container+'-repair'
    state = {'state': 'waiting', 'waiting_for': original['Id'], 'image': args.image}
    path = args.data_dir/'papers2_repair_watch.json'
    def save():
        state['updated'] = time.time()
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2)); temp.replace(path)
        print(json.dumps(state), flush=True)
    save()
    while True:
        current = inspect(args.container)
        if current['Id'] != original['Id']:
            raise RuntimeError('Converter was replaced externally; inspect before scheduling repair.')
        if not current['State']['Running']:
            break
        time.sleep(30)
    existing = subprocess.check_output(['docker', 'ps', '-a', '--filter', f'name=^/{name}$',
                                        '--format', '{{.State}}'], text=True).strip()
    if existing != 'running':
        if existing:
            subprocess.run(['docker', 'rm', name], check=True)
        subprocess.run(['docker', 'run', '-d', '--init', '--name', name, '--network', 'none',
                        '--user', original['Config']['User'], '--read-only', '--cap-drop', 'ALL',
                        '--security-opt', 'no-new-privileges', '--cpus', str(args.workers),
                        '--memory', '64g', '--tmpfs', '/tmp:rw,nosuid,nodev,size=1g',
                        '--mount', f'type=bind,src={args.data_dir},dst=/data',
                        '--mount', f'type=bind,src={args.data_dir}/papers2,dst=/data/papers2,readonly',
                        args.image, 'repair', '--apply', '--workers', str(args.workers),
                        '--timeout', str(args.timeout)], check=True)
    state.update(state='running', container_id=inspect(name)['Id']); save()
    code = subprocess.check_output(['docker', 'wait', state['container_id']], text=True).strip()
    state.update(state='finished', exit_code=int(code)); save()


if __name__ == '__main__':
    main()
