"""Wait for a specific conversion container, then resume once with a new image.

Runs on the conversion host under user systemd. Completed Markdown is reused.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def inspect(container):
    return json.loads(subprocess.check_output(['docker','inspect',container]))[0]


def launch_command(config, image, workers, timeout):
    cmd = list(config['docker_command'])
    for flag, value in [('--workers',workers),('--cpus',workers),('--timeout',timeout),('--stop-timeout',timeout+30)]:
        cmd[cmd.index(flag)+1] = str(value)
    cmd[cmd.index('convert')-1] = image
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--timeout',type=int,default=1800)
    args=p.parse_args()
    if args.workers<1 or args.timeout<1:
        p.error('Workers and timeout must be positive')
    config=json.loads(args.config.read_text())
    container=config['container']
    original=inspect(container)
    image=config['docker_command'][config['docker_command'].index('convert')-1]
    image_id=json.loads(subprocess.check_output(['docker','image','inspect',image]))[0]['Id']
    command=launch_command(config,image_id,args.workers,args.timeout)
    data=Path(config['data_dir'])
    record={'waiting_for':original['Id'],'retry_image':image_id,'workers':args.workers,'timeout':args.timeout,'state':'waiting'}
    state_path=data/'papers2_conversion_retry.json'
    def save():
        record['updated']=time.time()
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2));temp.replace(state_path)
        print(json.dumps(record),flush=True)
    save()
    while True:
        current=inspect(container)
        if current['Id']!=original['Id']:
            raise RuntimeError('Converter was replaced externally; refusing to replace it again')
        if not current['State']['Running']:
            break
        time.sleep(30)
    logs=subprocess.run(['docker','logs',container],capture_output=True,check=True)
    (data/'conversion-before-ocr-retry.log').write_bytes(logs.stdout+logs.stderr)
    subprocess.run(['docker','rm',container],check=True)
    completed=subprocess.run(command,text=True,capture_output=True,check=True)
    record.update(state='retry_started',container_id=completed.stdout.strip());save()
    code=subprocess.check_output(['docker','wait',record['container_id']],text=True).strip()
    record.update(state='retry_finished',exit_code=int(code));save()


if __name__=='__main__':
    main()
