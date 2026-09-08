"""Move a pre-rename data directory to the new one, safely.

The app works fine without this: if ``Documents\\Aether`` does not exist it
keeps using ``Documents\\RTLBaseline``. Run this only if you want the folder
renamed to match.

Sessions store an **absolute** path to their .npz file, so moving the folder
without rewriting those paths would leave every saved scan unloadable. This
does both, in the right order, and verifies before committing.

    python tools/migrate_data.py            # show what would happen
    python tools/migrate_data.py --apply
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aether import config


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate the data directory.")
    ap.add_argument("--apply", action="store_true",
                    help="actually move (default is a dry run)")
    ap.add_argument("--source", type=Path,
                    default=Path.home() / "Documents" / config.LEGACY_APP_NAME)
    ap.add_argument("--dest", type=Path,
                    default=Path.home() / "Documents" / config.APP_NAME)
    args = ap.parse_args()

    src, dst = args.source, args.dest
    print("from: %s" % src)
    print("to:   %s" % dst)

    if not src.is_dir():
        print("\nNothing to migrate -- the old directory does not exist.")
        return 0
    if dst.exists():
        print("\n%s already exists. Merge it by hand, or pass --dest." % dst)
        return 1

    db_path = src / "sessions.db"
    rows = []
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, npz_path FROM sessions WHERE npz_path != ''"
            ).fetchall()
        finally:
            conn.close()

    remapped = []
    for row in rows:
        old = Path(row["npz_path"])
        try:
            new = dst / old.relative_to(src)
        except ValueError:
            # Points outside the data directory; leave it alone.
            print("  session %s: path outside the folder, left as-is" % row["id"])
            continue
        remapped.append((row["id"], str(old), str(new)))

    print("\n%d session(s), %d spectrum path(s) to rewrite."
          % (len(rows), len(remapped)))
    if not args.apply:
        for sid, old, new in remapped[:5]:
            print("  #%s\n    %s\n -> %s" % (sid, old, new))
        if len(remapped) > 5:
            print("  ... and %d more" % (len(remapped) - 5))
        print("\nDry run. Re-run with --apply to perform the move.")
        return 0

    print("\nMoving...")
    shutil.move(str(src), str(dst))

    new_db = dst / "sessions.db"
    if remapped and new_db.exists():
        conn = sqlite3.connect(new_db)
        try:
            conn.executemany(
                "UPDATE sessions SET npz_path = ? WHERE id = ?",
                [(new, sid) for sid, _old, new in remapped],
            )
            conn.commit()
        finally:
            conn.close()
        print("Rewrote %d spectrum path(s)." % len(remapped))

    missing = [n for _s, _o, n in remapped if not Path(n).exists()]
    if missing:
        print("\nWARNING: %d spectrum file(s) are not where the database now "
              "expects them:" % len(missing))
        for m in missing[:5]:
            print("  " + m)
        return 2

    print("\nDone. Every spectrum file is where the database expects it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
