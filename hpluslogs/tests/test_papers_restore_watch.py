import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hpluslogs.scripts import watch_papers_restore as watcher


class RestoreWatchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_only_matching_active_restore_blocks_handoff(self):
        proc = self.root/'proc'; proc.mkdir()
        target = self.root/'restic-staging'
        for pid, path, state in [(1, str(target), 'S'), (2, '/other', 'S'), (3, str(target), 'Z')]:
            directory = proc/str(pid); directory.mkdir()
            (directory/'cmdline').write_bytes(b'\0'.join(x.encode() for x in ['restic','restore','snapshot','--target',path])+b'\0')
            (directory/'stat').write_text(f'{pid} (restic) {state} 0 0')
        self.assertEqual(watcher.restore_processes(target, proc), [1])

    def test_source_checksum_and_path_verification(self):
        (self.root/'paper.pdf').write_bytes(b'PDF')
        entry = {'path':'paper.pdf','size':3,'sha256':hashlib.sha256(b'PDF').hexdigest()}
        watcher.verify_sources(self.root, [entry])
        (self.root/'paper.pdf').write_bytes(b'BAD')
        with self.assertRaisesRegex(ValueError, 'checksum differs'):
            watcher.verify_sources(self.root, [entry])
        with self.assertRaisesRegex(ValueError, 'unsafe path'):
            watcher.verify_sources(self.root, [{'path':'../secret','size':0,'sha256':''}])

    def test_promotion_preserves_partial_directory_and_is_repeatable(self):
        stage = self.root/'papers2-restic'; stage.mkdir(); (stage/'restored.pdf').write_text('restored')
        destination = self.root/'papers2'; destination.mkdir(); (destination/'partial.pdf').write_text('partial')
        watcher.promote(stage,destination)
        self.assertEqual((destination/'restored.pdf').read_text(),'restored')
        backups = list(self.root.glob('papers2-transfer-partial-*'))
        self.assertEqual((backups[0]/'partial.pdf').read_text(),'partial')
        watcher.promote(stage,destination)
        self.assertEqual(len(list(self.root.glob('papers2-transfer-partial-*'))),1)

    def test_bad_restore_never_starts_container(self):
        stage = self.root/'papers2-restic'; stage.mkdir()
        ready = self.root/'seed.ready'; ready.touch()
        manifest = self.root/'source.json'
        manifest.write_text(json.dumps([{'path':'missing.pdf','size':3,'sha256':'bad'}]))
        config = {'data_dir':str(self.root),'staging':str(stage),'seed_ready':str(ready),
                  'source_manifest':str(manifest),'container':'test','workers':48,'docker_command':['docker','run']}
        with patch.object(watcher,'restore_processes',return_value=[]), \
             patch.object(watcher.subprocess,'run',return_value=SimpleNamespace(stdout='')) as run:
            with self.assertRaisesRegex(ValueError,'missing'):
                watcher.watch(config,lambda *args,**kwargs:None,0)
        self.assertEqual(run.call_count,1)
        self.assertNotIn('run',run.call_args.args[0])
        self.assertTrue(stage.exists())

    def test_valid_restore_preserves_ready_markdown_and_starts_once(self):
        stage=self.root/'papers2-restic';stage.mkdir();(stage/'paper.pdf').write_bytes(b'PDF')
        ready=self.root/'seed.ready';ready.touch()
        manifest=self.root/'source.json'
        manifest.write_text(json.dumps([{'path':'paper.pdf','size':3,'sha256':hashlib.sha256(b'PDF').hexdigest()}]))
        output=self.root/'papers2_markdown';output.mkdir();(output/'paper.md').write_bytes(b'Markdown')
        with sqlite3.connect(self.root/'papers2_markdown.sqlite3') as db:
            db.execute('create table conversions (markdown_path,markdown_sha256,status)')
            db.execute('insert into conversions values (?,?,?)',('paper.md',hashlib.sha256(b'Markdown').hexdigest(),'ready'))
        config={'data_dir':str(self.root),'staging':str(stage),'seed_ready':str(ready),
                'source_manifest':str(manifest),'container':'test','workers':48,'docker_command':['docker','run','-d','image']}
        states=[]
        results=[SimpleNamespace(stdout=''),SimpleNamespace(stdout='container-id'),SimpleNamespace(stdout='{"Running":true,"ExitCode":0}')]
        with patch.object(watcher,'restore_processes',return_value=[]), \
             patch.object(watcher.time,'sleep'),patch.object(watcher.subprocess,'run',side_effect=results):
            watcher.watch(config,lambda state,**extra:states.append((state,extra)))
        self.assertEqual(states[-1][0],'conversion_started')
        self.assertEqual(states[-1][1]['preserved_markdown'],1)
        self.assertTrue((self.root/'papers2/paper.pdf').exists())
