"""Command line: `urbanheat <command>`. Run from the repository root (config paths are relative to it)."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from urbanheat.config import load_config
from urbanheat.data import catalog, drive


def handover_filter(sensors):
    """skip(path) for the Drive walk: keep FINAL-DATASET/ and <SENSOR>-LANDSAT/PASSED-PAIRS/ of `sensors`."""
    def skip(path):
        parts = path.split("/")
        if "_to_delete" in parts:
            return True
        if parts[0] == "FINAL-DATASET":
            return False
        if parts[0].endswith("-LANDSAT"):
            if parts[0].split("-")[0] not in sensors:
                return True
            return len(parts) > 1 and parts[1] != "PASSED-PAIRS"
        return True
    return skip


def pair_of(path):
    """(sensor, pair folder) of a PASSED-PAIRS file path, else None."""
    parts = path.split("/")
    if len(parts) >= 4 and parts[1] == "PASSED-PAIRS":
        return parts[0].split("-")[0], parts[2]
    return None


def cmd_download(cfg, args):
    sensors = [s.upper() for s in (args.sensors or cfg["data"]["sensors"])]
    raw = Path(cfg["data"]["raw_dir"])
    print(f"listing Drive folder {cfg['data']['drive_root_id']} for {', '.join(sensors)} ...")
    entries = drive.walk(cfg["data"]["drive_root_id"], skip=handover_filter(sensors))
    files = [e for e in entries if not e.is_folder]

    if args.limit:
        keep = {}
        for e in files:
            p = pair_of(e.path)
            if p:
                keep.setdefault(p[0], [])
                if p[1] not in keep[p[0]] and len(keep[p[0]]) < args.limit:
                    keep[p[0]].append(p[1])
        files = [e for e in files if pair_of(e.path) is None or pair_of(e.path)[1] in keep[pair_of(e.path)[0]]]

    pairs = Counter(pair_of(e.path)[0] for e in files if pair_of(e.path) and e.name.startswith("pair_info"))
    todo = [e for e in files if not (raw / e.path).exists()]
    print(f"{len(files)} files ({dict(pairs)} pairs); {len(todo)} not on disk yet")
    if args.dry_run:
        for e in todo[:20]:
            print("  would fetch", e.path)
        return

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda e: (e.path, drive.download(e.id, raw / e.path)), todo))
    failed = [p for p, r in results if r == "FAIL"]
    print(f"downloaded {sum(r == 'ok' for _, r in results)}, failed {len(failed)}")
    for p in failed:
        print("  FAIL", p)


def cmd_catalog(cfg, args):
    idx = catalog.load_index(cfg["data"]["index_file"], cfg["data"]["raw_dir"], cfg["data"]["sensors"])
    catalog.check_split(idx, cfg["split"]["column"])
    print(catalog.summary(idx, cfg["split"]["column"]).to_string())
    missing = idx[~idx.on_disk]
    print(f"\non disk: {int(idx.on_disk.sum())}/{len(idx)} pairs")
    if len(missing):
        print("missing (run `urbanheat download`):", ", ".join(missing.pair_id[:15]), "..." if len(missing) > 15 else "")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="urbanheat")
    ap.add_argument("--config", help="YAML file overriding configs/default.yaml")
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="fetch Member 1's handover (FINAL-DATASET + PASSED-PAIRS) from Drive")
    d.add_argument("--sensors", nargs="+", help="subset of EMIT ENMAP HYPERION (default: config)")
    d.add_argument("--limit", type=int, help="only the first N pairs per sensor (quick test)")
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    d.set_defaults(func=cmd_download)

    c = sub.add_parser("catalog", help="pairs per sensor and split, and what is on disk")
    c.set_defaults(func=cmd_catalog)

    args = ap.parse_args(argv)
    args.func(load_config(args.config), args)


if __name__ == "__main__":
    main()
