"""Read-only SHA256 verification of frozen B02/G01/Round01 and old CXR source."""
import argparse
import json
import re
from pathlib import Path

from src.utils import sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--old-root',help='Existing chest X-ray project root; read only')
    args=p.parse_args()
    frozen=json.loads(Path('outputs/migration_review/frozen_inventory.json').read_text(encoding='utf-8'))
    bad=[name for name,digest in frozen.items() if not Path(name).exists() or sha256(name)!=digest]
    report={'frozen_count':len(frozen),'frozen_changed':bad}
    if args.old_root:
        # The historical inventory's root string has a malformed escape; parse its file rows safely.
        raw=Path('outputs/migration_review/source_inventory.json').read_text(encoding='utf-8')
        rows=dict(re.findall(r'"([^"\n]+)": "([0-9a-f]{64})"',raw))
        root=Path(args.old_root)
        report.update(old_source_count=len(rows),old_source_changed=[name for name,digest in rows.items()
                      if not (root/name).exists() or sha256(root/name)!=digest])
    print(json.dumps(report,ensure_ascii=False))
    if bad or report.get('old_source_changed'):raise SystemExit(1)


if __name__=='__main__':main()
