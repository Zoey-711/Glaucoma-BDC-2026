"""Freeze prior experiment artifacts before new experiments, without modifying them."""
import json
from pathlib import Path
from src.utils import sha256, write_json

INVENTORY=Path('outputs/M02d_L03/frozen_inventory.json')


def freeze_or_verify():
    if INVENTORY.exists():
        inventory=json.loads(INVENTORY.read_text(encoding='utf-8'))
    else:
        inventory=json.loads(Path('outputs/migration_review/frozen_inventory.json').read_text(encoding='utf-8'))
        for root in ['results/G02','results/M02a','results/M02b','results/M02c','outputs/M02']:
            for path in Path(root).rglob('*'):
                if path.is_file(): inventory[str(path)]=sha256(path)
        write_json(INVENTORY,inventory)
    bad=[name for name,digest in inventory.items() if not Path(name).is_file() or sha256(name)!=digest]
    if bad: raise ValueError(f'Frozen assets changed: {bad}')
    return len(inventory)


def checksums(directory):
    root=Path(directory)
    files=sorted(p for p in root.iterdir() if p.is_file() and p.name!='checksums.sha256')
    (root/'checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in files),encoding='utf-8')
