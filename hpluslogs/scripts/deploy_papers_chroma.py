"""Deploy the local paper index beside an existing conversion deployment."""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hpluslogs.services.papers_remote import Remote


def deploy(remote, port=18081, workers=4):
    project = Path(__file__).resolve().parents[1]
    name = remote.name+'-chroma'
    image = name+':latest'
    remote.command('mkdir','-p',remote.path+'/chroma-build',remote.path+'/data/chroma')
    with tempfile.TemporaryDirectory(prefix='papers-chroma-build-') as tmp:
        root = Path(tmp)
        for relative in ['papers_chroma_cli.py','services/papers_chroma.py','services/paper_conversion_worker.py']:
            target = root/'hpluslogs'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(project/relative,target)
        for filename in ['Dockerfile','requirements.txt']:
            shutil.copy2(project/'docker/papers-chroma'/filename,root/filename)
        remote.rsync(str(root)+'/',remote.location('chroma-build')+'/', '--delete')
    remote.command('docker','build','-t',image,remote.path+'/chroma-build')
    uid = remote.command('id','-u',capture=True).stdout.strip()
    gid = remote.command('id','-g',capture=True).stdout.strip()
    common = ['--network','host','--user',f'{uid}:{gid}','--cap-drop','ALL',
              '--security-opt','no-new-privileges','--restart','unless-stopped']
    def state(container):
        return remote.command('docker','ps','-a','--filter',f'name=^/{container}$','--format','{{.State}}',capture=True).stdout.strip()
    current = state(name)
    if not current:
        remote.command('docker','run','-d','--name',name,*common,'--cpus','4','--memory','32g',
            '--mount',f'type=bind,src={remote.path}/data/chroma,dst=/chroma','--entrypoint','chroma',image,
            'run','--host','127.0.0.1','--port',str(port),'--path','/chroma')
    elif current != 'running':
        remote.command('docker','start',name)
    # Readiness gate before launching/resuming the indexer.
    remote.command('docker','exec',name,'python','-c',
        "import time,urllib.request\nfor i in range(60):\n try:\n  urllib.request.urlopen('http://127.0.0.1:%d/api/v2/heartbeat'); break\n except Exception:\n  time.sleep(1)\nelse: raise SystemExit('Chroma failed readiness')" % port)
    worker = name+'-index'
    if state(worker):
        remote.command('docker','stop','--timeout','30',worker)
        remote.command('docker','rm',worker)
    remote.command('docker','run','-d','--init','--name',worker,*common,'--cpus',str(workers),
        '--memory','16g','--mount',f'type=bind,src={remote.path}/data,dst=/data',image,
        'papers-chroma-index','--chroma-port',str(port),'--workers',str(workers),'--watch')
    print(f'Chroma: {name}; continuous indexer: {worker}; loopback port: {port}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',required=True)
    parser.add_argument('--user',required=True)
    parser.add_argument('--path',required=True)
    parser.add_argument('--port',type=int,default=18081)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.workers <= 64:
        parser.error('Invalid port or worker count')
    deploy(Remote(args.host,args.user,args.path),args.port,args.workers)
