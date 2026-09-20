"""Remote conversion commands on the existing flat CLI."""
import subprocess
import click
from hpluslogs.services import papers_remote


def connection_options(fn):
    for name in ('path', 'user', 'host'):
        fn = click.option('--' + name, required=True)(fn)
    return fn


def register(cli):
    def invoke(action, obj, host, user, path, **kwargs):
        remote = papers_remote.Remote(host, user, path)
        try:
            if action in ('deploy', 'pull', 'watch_restore'):
                return getattr(papers_remote, action)(obj['data_dir'], remote, **kwargs)
            return getattr(papers_remote, action)(remote, **kwargs)
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            raise click.ClickException(str(error)) from error

    @cli.command('papers-remote-build')
    @connection_options
    @click.pass_obj
    def build(obj, **kwargs):
        """Build an updated conversion image; preserve running jobs and remote data."""
        invoke('build', obj, **kwargs)

    @cli.command('papers-remote-deploy')
    @connection_options
    @click.option('--workers', type=click.IntRange(min=1), default=48, show_default=True)
    @click.option('--timeout', type=click.IntRange(min=1), default=600)
    @click.option('--transfer', type=click.Choice(['tar', 'rsync']), default='tar', show_default=True)
    @click.pass_obj
    def deploy(obj, **kwargs):
        """Copy PDFs/checkpoints, build Docker, and resume remote conversion only."""
        invoke('deploy', obj, **kwargs)

    @cli.command('papers-remote-progress')
    @connection_options
    @click.option('--days', type=click.IntRange(min=1), default=30)
    @click.option('--searches', type=click.IntRange(min=0), default=0)
    @click.option('--index-multiplier', type=click.FloatRange(min=0), default=1.0)
    @click.pass_obj
    def progress(obj, **kwargs):
        """Read remote conversion progress, ETA and monthly cost projections."""
        invoke('progress', obj, **kwargs)

    @cli.command('papers-remote-stop')
    @connection_options
    @click.pass_obj
    def stop(obj, **kwargs):
        """Stop scheduling PDFs; finish active conversions and save progress."""
        invoke('stop', obj, **kwargs)

    @cli.command('papers-remote-pull')
    @connection_options
    @click.pass_obj
    def pull(obj, **kwargs):
        """Pull Markdown/checkpoint from a stopped container; back up local state."""
        invoke('pull', obj, **kwargs)

    @cli.command('papers-remote-watch-restore')
    @connection_options
    @click.option('--staging-name', default='papers2-restic', show_default=True)
    @click.option('--workers', type=click.IntRange(min=1), default=48)
    @click.option('--timeout', type=click.IntRange(min=1), default=600)
    @click.pass_obj
    def watch_restore(obj, **kwargs):
        """Watch a remote restic restore, preserve local progress, then start conversion."""
        invoke('watch_restore', obj, **kwargs)

    @cli.command('papers-remote-monitor')
    @connection_options
    @click.option('--interval', type=click.IntRange(min=10), default=300, show_default=True)
    @click.pass_obj
    def monitor(obj, **kwargs):
        """Record progress/ETA every five minutes on the remote host until conversion ends."""
        invoke('monitor', obj, **kwargs)

    @cli.command('papers-remote-repair')
    @connection_options
    @click.option('--workers', type=click.IntRange(min=1), default=16)
    @click.option('--timeout', type=click.IntRange(min=1), default=1800)
    @click.pass_obj
    def repair(obj, **kwargs):
        """Schedule offline OCR repair after the current conversion finishes."""
        invoke('repair', obj, **kwargs)
