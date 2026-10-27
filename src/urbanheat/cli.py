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


def cmd_harmonize(cfg, args):
    from urbanheat.harmonize.build import build

    out, table = build(cfg, workers=args.workers)
    print(f"\nwrote {out} ({out.stat().st_size / 1e9:.2f} GB)")
    print(table.groupby(["sensor", "split"]).agg(pairs=("pair_id", "size"), valid_px=("n_valid", "sum"))
          .unstack(fill_value=0).to_string())
    empty = table[table.n_valid == 0]
    if len(empty):
        print(f"{len(empty)} pairs have no valid pixel at {cfg['harmonize']['grid_m']} m:", ", ".join(empty.pair_id))


def cmd_change(cfg, args):
    import ee
    import pandas as pd
    from urbanheat.change import gee

    if not cfg["ee"]["project"]:
        raise SystemExit("set ee.project in configs/local.yaml (your Earth Engine cloud project)")
    ee.Initialize(project=cfg["ee"]["project"])
    ccfg = cfg["change"]
    years = list(range(ccfg["years"][0], ccfg["years"][1] + 1))
    idx = pd.read_csv(cfg["data"]["index_file"])
    cells = idx.groupby("cell_id").agg(lon=("cell_lon", "first"), lat=("cell_lat", "first"),
                                       climate_group=("climate_group", "first")).reset_index()
    if args.cells:
        cells = cells[cells.cell_id.isin(args.cells)]
    if args.limit:
        cells = cells.head(args.limit)
    if args.transitions:
        return change_transitions(ee, gee, cfg, cells, years, args)
    out = Path(cfg["data"]["derived_dir"]) / "change" / "cell_year.csv"
    done = pd.read_csv(out) if out.exists() and not args.overwrite else pd.DataFrame(columns=["cell_id"])
    todo = cells[~cells.cell_id.isin(done.cell_id)]
    print(f"{len(cells)} cells, {len(todo)} to do, years {years[0]}-{years[-1]}")

    def run(c):
        rows, start = gee.cell_series(ee, c.lon, c.lat, years, ccfg)
        return [dict(cell_id=c.cell_id, climate_group=c.climate_group, hot_season_start_month=start, **r) for r in rows]

    new, failed = [], []
    with ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(run, c): c.cell_id for c in todo.itertuples()}
        for j, f in enumerate(futures, 1):
            try:
                new += f.result()
            except Exception as e:
                failed.append(futures[f])
                print(f"  FAIL {futures[f]}: {str(e)[:200]}")
            if j % 20 == 0 or j == len(futures):
                print(f"{j}/{len(futures)} cells")
                if new:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    pd.concat([done, pd.DataFrame(new)]).to_csv(out, index=False)
    t = pd.concat([done, pd.DataFrame(new)])
    if len(t):
        t["suhi_c"] = t.lst_built_c - t.lst_rural_c
        out.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(out, index=False)
    print(f"failed: {failed or 'none'} | {out}")


def change_transitions(ee, gee, cfg, cells, years, args):
    """LST change per land-cover transition per cell, first 3 vs last 3 years (needs cell_year.csv for the season)."""
    import pandas as pd

    root = Path(cfg["data"]["derived_dir"]) / "change"
    season = pd.read_csv(root / "cell_year.csv").groupby("cell_id").hot_season_start_month.first()
    cells = cells[cells.cell_id.isin(season.dropna().index)]
    early, late = years[:3], years[-3:]

    def run(c):
        r = gee.transition_warming(ee, c.lon, c.lat, int(season[c.cell_id]), early, late, cfg["change"])
        return dict(cell_id=c.cell_id, climate_group=c.climate_group, early=f"{early[0]}-{early[-1]}",
                    late=f"{late[0]}-{late[-1]}", **r)

    rows, failed = [], []
    with ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(run, c): c.cell_id for c in cells.itertuples()}
        for j, f in enumerate(futures, 1):
            try:
                rows.append(f.result())
            except Exception as e:
                failed.append(futures[f])
                print(f"  FAIL {futures[f]}: {str(e)[:200]}")
            if j % 50 == 0 or j == len(futures):
                print(f"{j}/{len(futures)} cells")
    pd.DataFrame(rows).to_csv(root / "transitions.csv", index=False)
    print(f"failed: {failed or 'none'} | {root / 'transitions.csv'}")


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

    h = sub.add_parser("harmonize", help="build the common dataset (all sensors, one band set, one grid)")
    h.add_argument("--workers", type=int, default=4)
    h.set_defaults(func=cmd_harmonize)

    ch = sub.add_parser("change", help="green / built / water area and heat island per cell and year (Earth Engine)")
    ch.add_argument("--cells", nargs="+", help="only these cell_ids")
    ch.add_argument("--limit", type=int, help="only the first N cells (quick test)")
    ch.add_argument("--workers", type=int, default=6)
    ch.add_argument("--overwrite", action="store_true", help="recompute cells already done")
    ch.add_argument("--transitions", action="store_true",
                    help="LST change per land-cover transition (green->built, ...), first vs last 3 years")
    ch.set_defaults(func=cmd_change)

    args = ap.parse_args(argv)
    args.func(load_config(args.config), args)


if __name__ == "__main__":
    main()
