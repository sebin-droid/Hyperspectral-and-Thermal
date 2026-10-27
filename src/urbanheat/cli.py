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


def cmd_thermal_context(cfg, args):
    import ee
    import pandas as pd
    from urbanheat.data import thermal_context as tc

    if not cfg["ee"]["project"]:
        raise SystemExit("set ee.project in configs/local.yaml (your Earth Engine cloud project)")
    ee.Initialize(project=cfg["ee"]["project"])
    tcfg = cfg["thermal_context"]
    idx = catalog.load_index(cfg["data"]["index_file"], cfg["data"]["raw_dir"],
                             [s.upper() for s in (args.sensors or cfg["data"]["sensors"])])
    idx = idx[idx.on_disk]
    if args.limit:
        idx = idx.groupby("sensor").head(args.limit)
    out_root = Path(cfg["data"]["derived_dir"]) / "thermal_context"
    summary_file = out_root / "summary.csv"
    done = pd.read_csv(summary_file).set_index("pair_id") if summary_file.exists() else pd.DataFrame()

    def run(row):
        out = out_root / row.rel_dir / "thermal_context.tif"
        if out.exists() and not args.overwrite and row.pair_id in done.index:
            return done.loc[row.pair_id].to_dict() | {"pair_id": row.pair_id}
        grid, profile = tc.pair_grid(row.landsat_path)
        b = rasterio_bounds(profile)
        region = ee.Geometry.Rectangle(list(b), proj=grid["crsCode"], geodesic=False)
        date = tc.landsat_date(row.landsat_scene)
        image, n_scenes = tc.context_image(ee, region, date, tcfg)
        rec = dict(pair_id=row.pair_id, sensor=row.sensor, landsat_date=str(date), n_scenes=n_scenes.getInfo())
        if rec["n_scenes"] == 0:
            return rec | dict(coverage=0.0)
        data = tc.fetch(ee, image, grid, float(row.res_m))
        tc.write(out, data, profile)
        return rec | tc.qc(data, row.landsat_path)

    rows, failed = [], []
    with ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(run, r): r.pair_id for r in idx.itertuples()}
        for j, f in enumerate(futures, 1):
            try:
                rows.append(f.result())
            except Exception as e:
                failed.append(futures[f])
                print(f"  FAIL {futures[f]}: {str(e)[:200]}")
            if j % 25 == 0 or j == len(futures):
                print(f"{j}/{len(futures)} pairs")

    s = pd.DataFrame(rows)
    if len(done):
        s = pd.concat([done.reset_index()[~done.index.isin(s.pair_id)], s])
    out_root.mkdir(parents=True, exist_ok=True)
    s.sort_values("pair_id").to_csv(summary_file, index=False)
    print(s.groupby("sensor")[["n_scenes", "median_n_obs", "coverage", "r_with_target"]].median().round(2).to_string())
    print(f"failed: {failed or 'none'}  |  summary: {summary_file}")


def rasterio_bounds(profile):
    t, w, h = profile["transform"], profile["width"], profile["height"]
    xs, ys = (t.c, t.c + t.a * w), (t.f, t.f + t.e * h)
    return min(xs), min(ys), max(xs), max(ys)


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

    t = sub.add_parser("thermal-context", help="median LST of other dates per pair, from Earth Engine")
    t.add_argument("--sensors", nargs="+", help="subset of EMIT ENMAP HYPERION (default: config)")
    t.add_argument("--limit", type=int, help="only the first N pairs per sensor (quick test)")
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--overwrite", action="store_true", help="recompute pairs already done")
    t.set_defaults(func=cmd_thermal_context)

    args = ap.parse_args(argv)
    args.func(load_config(args.config), args)


if __name__ == "__main__":
    main()
