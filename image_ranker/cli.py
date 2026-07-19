from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from . import launchd
from .config import Settings
from .db import Database
from .ingest import InvalidImage, ingest_file
from .jobs import JobPolicy, crawl_with_latest_model, run_once, watch
from .ml import train
from .server import serve


def main() -> None:
    parser = argparse.ArgumentParser(prog="image-ranker")
    sub = parser.add_subparsers(dest="command", required=True)
    server = sub.add_parser("serve")
    server.add_argument("--remote", action="store_true", help="serve the authenticated remote API on port 8788")
    add = sub.add_parser("import"); add.add_argument("paths", nargs="+", type=Path)
    seed = sub.add_parser("seed"); seed.add_argument("--limit", type=int, default=60)
    fit = sub.add_parser("train"); fit.add_argument("--epochs", type=int, default=300)
    jobs = sub.add_parser("jobs", help="run local training and discovery jobs")
    jobs.add_argument("--watch", action="store_true", help="repeat until interrupted")
    jobs.add_argument("--interval", type=float, help="seconds between watched passes")
    jobs.add_argument("--train-minimum", type=int)
    jobs.add_argument("--train-batch", type=int)
    jobs.add_argument("--unranked-threshold", type=int)
    jobs.add_argument("--crawl-limit", type=int)
    jobs.add_argument("--epochs", type=int)
    launchd.add_cli_parser(sub)
    args = parser.parse_args()
    settings = Settings.load()
    if args.command == "launchd":
        launchd.run_cli(args, settings)
        return
    settings.ensure()
    if args.command == "serve":
        serve(settings, remote=args.remote)
        return
    db = Database(settings.database)
    db.initialize()
    if args.command == "import":
        imported = 0
        for path in args.paths:
            try: ingest_file(db, settings.images, path); imported += 1
            except InvalidImage as exc: print(f"Skipped {path}: {exc}")
        print(f"Imported {imported} images")
    elif args.command == "seed":
        print(json.dumps(crawl_with_latest_model(settings, db, args.limit), sort_keys=True))
    elif args.command == "train": print(train(settings.database, settings.images, settings.models, args.epochs))
    elif args.command == "jobs":
        policy = JobPolicy.load()
        overrides = {
            field: value
            for field, value in {
                "interval_seconds": args.interval,
                "train_minimum": args.train_minimum,
                "train_batch": args.train_batch,
                "unranked_threshold": args.unranked_threshold,
                "crawl_limit": args.crawl_limit,
                "epochs": args.epochs,
            }.items()
            if value is not None
        }
        policy = replace(policy, **overrides)
        if args.watch:
            try:
                for report in watch(settings, policy):
                    print(json.dumps(report, sort_keys=True), flush=True)
            except KeyboardInterrupt:
                print("Stopped local jobs")
        else:
            print(json.dumps(run_once(settings, policy), sort_keys=True))


if __name__ == "__main__": main()
