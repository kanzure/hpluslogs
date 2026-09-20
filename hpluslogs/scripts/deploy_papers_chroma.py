"""Deploy the local paper index beside an existing conversion deployment."""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hpluslogs.services.papers_remote import Remote


def deploy(remote, port=18081, concurrency=80, batch_size=1000, cost_limit=5.0,
           chroma_data_path=None, env_file=None, prepare_only=False):
    project = Path(__file__).resolve().parents[1]
    name = remote.name+'-chroma'
    image = name+':latest'
    storage = Remote(remote.target.split('@')[1],remote.target.split('@')[0],
                     chroma_data_path or remote.path+'/data/chroma').path
    remote.command('mkdir','-p',remote.path+'/chroma-build',storage)
    if not prepare_only:
        if not env_file:
            raise ValueError('Supply --env-file with a private remote OPENROUTER_API_KEY file')
        remote.command('test','-f',env_file)
    with tempfile.TemporaryDirectory(prefix='papers-chroma-build-') as tmp:
        root = Path(tmp)
        for relative in ['papers_chroma_cli.py','services/papers_chroma.py','services/paper_embeddings.py','integrations/openrouter.py','core/paper_prompts.py']:
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
        remote.command('docker','run','-d','--name',name,*common,'--cpus','16','--memory','96g',
            '--mount',f'type=bind,src={storage},dst=/chroma','--entrypoint','chroma',image,
            'run','--host','127.0.0.1','--port',str(port),'--path','/chroma')
    else:
        import json
        mounts=json.loads(remote.command('docker','inspect','--format','{{json .Mounts}}',name,capture=True).stdout)
        actual=next(m['Source'] for m in mounts if m['Destination']=='/chroma')
        if actual != storage:
            raise ValueError('Existing Chroma storage differs; explicitly migrate or retire the existing server first')
        remote.command('docker','update','--cpus','16','--memory','96g',name)
        if current != 'running':
            remote.command('docker','start',name)
    # Readiness gate before launching/resuming the indexer.
    remote.command('docker','exec',name,'python','-c',
        "import time,urllib.request\nfor i in range(60):\n try:\n  urllib.request.urlopen('http://127.0.0.1:%d/api/v2/heartbeat'); break\n except Exception:\n  time.sleep(1)\nelse: raise SystemExit('Chroma failed readiness')" % port)
    if prepare_only:
        print(f'Prepared Chroma at {storage}; no embedding requests started.')
        return
    worker = name+'-index'
    if state(worker):
        remote.command('docker','stop','--timeout','30',worker)
        remote.command('docker','rm',worker)
    remote.command('docker','run','-d','--init','--name',worker,*common,'--cpus','8',
        '--memory','64g','--env-file',env_file,'--mount',f'type=bind,src={remote.path}/data,dst=/data',image,
        'papers-chroma-index','--chroma-port',str(port),'--concurrency',str(concurrency),
        '--batch-size',str(batch_size),'--cost-limit',str(cost_limit),'--watch')
    print(f'Chroma: {name}; continuous indexer: {worker}; loopback port: {port}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',required=True)
    parser.add_argument('--user',required=True)
    parser.add_argument('--path',required=True)
    parser.add_argument('--port',type=int,default=18081)
    parser.add_argument('--concurrency','--workers',type=int,default=80)
    parser.add_argument('--batch-size',type=int,default=1000)
    parser.add_argument('--cost-limit',type=float,default=5.0)
    parser.add_argument('--chroma-data-path')
    parser.add_argument('--env-file',help='Private remote file containing OPENROUTER_API_KEY')
    parser.add_argument('--prepare-only',action='store_true',help='Build and prepare server without starting paid embeddings')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.concurrency <= 256 or not 1 <= args.batch_size <= 1000 or not args.cost_limit > 0:
        parser.error('Invalid port, concurrency, batch size or cost limit')
    deploy(Remote(args.host,args.user,args.path),args.port,args.concurrency,args.batch_size,args.cost_limit,
           args.chroma_data_path,args.env_file,args.prepare_only)
