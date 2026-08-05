from __future__ import annotations

import argparse
import logging
from pathlib import Path

from glioma_shift_atlas.configuration import load_config
from glioma_shift_atlas.dataflow import load_manifest, manifest_fingerprint, stratified_assignments, write_assignments, write_data_registry


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="glioma-atlas")
    commands = root.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--config", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--destination", type=Path, required=True)
    for name in ("train", "pretrain", "finetune", "evaluate", "infer", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--cohort", type=str)
        command.add_argument("--destination", type=Path)
    return root


def inspect_command(config_path: Path) -> None:
    config = load_config(config_path)
    records = load_manifest(config.data.manifest)
    logging.info("patients=%d manifest=%s", len(records), manifest_fingerprint(config.data.manifest))


def prepare_command(manifest: Path, destination: Path) -> None:
    records = load_manifest(manifest)
    assignments = stratified_assignments(records, 5, 0.6, 0.2, 42)
    destination.mkdir(parents=True, exist_ok=True)
    write_assignments(assignments, destination / "splits.csv")
    write_data_registry(manifest, records, destination / "registry.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = parser().parse_args()
    if arguments.command == "inspect":
        inspect_command(arguments.config)
        return
    if arguments.command == "prepare":
        prepare_command(arguments.manifest, arguments.destination)
        return
    config = load_config(arguments.config)
    logging.info("command=%s cohorts=%s output=%s", arguments.command, config.data.cohorts, config.runtime.output)


if __name__ == "__main__":
    main()
