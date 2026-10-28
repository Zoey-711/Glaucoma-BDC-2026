"""SQLite registry: atomic claims, null measurements, immutable completed configs."""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


FIELDS = ['experiment_id', 'backbone', 'input_size', 'loss', 'sampler', 'augmentation',
          'optimizer', 'lr', 'seed', 'best_epoch', 'AUROC', 'AUPRC',
          'sensitivity_at_90_specificity', 'train_duration', 'model_size_mb',
          'local_latency_ms', 't4_latency_ms', 'checkpoint_path', 'config_hash', 'status']


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def training_hash(config):
    c = {k: v for k, v in config.items() if k not in ['experiment_id', 'output', 'plan', 'config_hash', 'bound_candidates', 'candidate']}
    return config_hash(c)


class Registry:
    def __init__(self, path='outputs/experiment_manager/registry.sqlite'):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, config_hash TEXT, training_hash TEXT, status TEXT, payload TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS events (time TEXT, id TEXT, status TEXT, detail TEXT)')

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def register(self, c):
        eid, digest = c['experiment_id'], config_hash(c)
        row = {k: None for k in FIELDS}
        row.update(experiment_id=eid, backbone=c['model']['name'], input_size=c['image_size'],
                   loss=c['loss']['name'], sampler=c['sampler'], augmentation=c['augmentation'],
                   optimizer='AdamW', lr=c['lr'], seed=c['seed'], config_hash=digest,
                   status=c.get('plan', {}).get('initial_status', 'planned'), config=c,
                   reason=c.get('plan', {}).get('reason'), output=c['output'])
        with self.connect() as db:
            old = db.execute('SELECT config_hash,status FROM experiments WHERE id=?', (eid,)).fetchone()
            if old:
                if old[0] != digest:
                    raise ValueError(f'{eid}: config changed; use an explicitly versioned experiment ID')
                return self.get(eid)
            db.execute('INSERT INTO experiments VALUES (?,?,?,?,?)', (eid, digest, training_hash(c), row['status'], json.dumps(row)))
        return row

    def get(self, eid):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM experiments WHERE id=?', (eid,)).fetchone()
        if not row:
            raise KeyError(eid)
        return json.loads(row[0])

    def rows(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT payload FROM experiments ORDER BY id')]

    def update(self, eid, **values):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = json.loads(db.execute('SELECT payload FROM experiments WHERE id=?', (eid,)).fetchone()[0])
            row.update(values)
            db.execute('UPDATE experiments SET status=?,payload=? WHERE id=?', (row['status'], json.dumps(row), eid))
            db.execute('INSERT INTO events VALUES (?,?,?,?)', (datetime.now(timezone.utc).isoformat(), eid, row['status'], json.dumps(values)))

    def claim(self, eid):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            raw = db.execute('SELECT training_hash,status,payload FROM experiments WHERE id=?', (eid,)).fetchone()
            if raw[1] in ['completed', 'reused']:
                return False
            if raw[1] not in ['planned', 'blocked']:
                raise ValueError(f'{eid}: cannot claim status {raw[1]}')
            duplicate = db.execute("SELECT id,payload FROM experiments WHERE training_hash=? AND status IN ('completed','reused','running') AND id!=?", (raw[0], eid)).fetchone()
            row = json.loads(raw[2])
            if duplicate:
                other = json.loads(duplicate[1])
                if other['status'] == 'running':
                    raise ValueError(f'Equivalent experiment already running: {duplicate[0]}')
                for k in FIELDS:
                    if k not in ['experiment_id', 'config_hash', 'status']:
                        row[k] = other.get(k)
                row.update(status='reused', reused_from=duplicate[0], output=other['output'])
                claimed = False
            else:
                row.update(status='running', reason=None)
                claimed = True
            db.execute('UPDATE experiments SET status=?,payload=? WHERE id=?', (row['status'], json.dumps(row), eid))
            return claimed

    def export(self):
        rows = self.rows()
        pd.DataFrame([{k: r.get(k) for k in FIELDS} for r in rows], columns=FIELDS).to_csv(self.path.parent / 'experiment_registry.csv', index=False)
