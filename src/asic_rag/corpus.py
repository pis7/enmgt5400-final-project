"""Corpus assembly for the ASIC design RAG system.

Walks an mflowgen build directory, collects useful text files (reports, logs,
RTL, constraints), tags each with metadata, and writes a JSONL corpus file.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Collection rules: (stage_prefix, filename_or_glob, report_type, tool)
# ---------------------------------------------------------------------------
STAGE_RULES: list[tuple[str, str, str, str]] = [
    # 01 - Verilator pickle
    ("01-verilator-pickle", "*_pickled.v", "rtl", "verilator"),
    ("01-verilator-pickle", "run.log", "log", "verilator"),
    # 03 - Synopsys DC synthesis
    ("03-synopsys-dc-synth", "timing.rpt", "timing-report", "synopsys-dc"),
    ("03-synopsys-dc-synth", "area.rpt", "area-report", "synopsys-dc"),
    ("03-synopsys-dc-synth", "resources.rpt", "resource-report", "synopsys-dc"),
    ("03-synopsys-dc-synth", "run.log", "log", "synopsys-dc"),
    ("03-synopsys-dc-synth", "post-synth.sdc", "constraints", "synopsys-dc"),
    # 05 - Cadence Innovus PnR
    ("05-cadence-innovus-pnr", "timing-setup.rpt", "timing-report", "cadence-innovus"),
    ("05-cadence-innovus-pnr", "timing-hold.rpt", "timing-report", "cadence-innovus"),
    ("05-cadence-innovus-pnr", "area.rpt", "area-report", "cadence-innovus"),
    ("05-cadence-innovus-pnr", "drc.rpt", "drc-report", "cadence-innovus"),
    ("05-cadence-innovus-pnr", "run.log", "log", "cadence-innovus"),
    # 06 - Synopsys PT STA
    ("06-synopsys-pt-sta", "timing-setup.rpt", "timing-report", "synopsys-pt"),
    ("06-synopsys-pt-sta", "timing-hold.rpt", "timing-report", "synopsys-pt"),
    ("06-synopsys-pt-sta", "timing-setup-summary.rpt", "timing-report", "synopsys-pt"),
    ("06-synopsys-pt-sta", "timing-hold-summary.rpt", "timing-report", "synopsys-pt"),
    ("06-synopsys-pt-sta", "run.log", "log", "synopsys-pt"),
    # 08 - Synopsys PT power
    ("08-synopsys-pt-pwr", "*-summary.rpt", "power-report", "synopsys-pt"),
    ("08-synopsys-pt-pwr", "*-detailed.rpt", "power-report", "synopsys-pt"),
    # 09 - Mentor Calibre DRC
    ("09-mentor-calibre-drc", "main-drc/run.summary", "drc-report", "mentor-calibre"),
    ("09-mentor-calibre-drc", "main-drc/run.log", "log", "mentor-calibre"),
    # 11 - Summarize results
    ("11-summarize-results", "run.log", "summary", "mflowgen"),
]


def discover_variants(build_dir: Path) -> list[Path]:
    """Find design variant directories (those containing a run-flow script)."""
    variants = []
    for child in sorted(build_dir.iterdir()):
        if child.is_dir() and (child / "run-flow").exists():
            variants.append(child)
    return variants


def collect_documents(build_dir: Path) -> list[dict]:
    """Collect all corpus documents from the build directory."""
    docs: list[dict] = []
    variants = discover_variants(build_dir)

    for variant_dir in variants:
        variant_name = variant_dir.name

        for stage_prefix, pattern, report_type, tool in STAGE_RULES:
            stage_dir = variant_dir / stage_prefix
            if not stage_dir.exists():
                continue

            # Pattern may contain a subdirectory (e.g. "main-drc/run.summary")
            matched = list(stage_dir.glob(pattern))
            for filepath in sorted(matched):
                if not filepath.is_file():
                    continue
                try:
                    content = filepath.read_text(errors="replace")
                except OSError:
                    continue

                doc_id = f"{variant_name}/{stage_prefix}/{filepath.relative_to(stage_dir)}"
                docs.append({
                    "id": doc_id,
                    "metadata": {
                        "source_path": str(filepath),
                        "design_variant": variant_name,
                        "flow_stage": stage_prefix,
                        "tool": tool,
                        "report_type": report_type,
                    },
                    "content": content,
                })

    # Collect build-root files: batch YAMLs and report.txt
    for yaml_path in sorted(build_dir.glob("batch-*.yml")):
        try:
            content = yaml_path.read_text()
        except OSError:
            continue
        docs.append({
            "id": f"build-config/{yaml_path.name}",
            "metadata": {
                "source_path": str(yaml_path),
                "design_variant": yaml_path.stem.removeprefix("batch-"),
                "flow_stage": "build-config",
                "tool": "mflowgen",
                "report_type": "build-config",
            },
            "content": content,
        })

    report_txt = build_dir / "report.txt"
    if report_txt.exists():
        docs.append({
            "id": "build-config/report.txt",
            "metadata": {
                "source_path": str(report_txt),
                "design_variant": "all",
                "flow_stage": "build-config",
                "tool": "mflowgen",
                "report_type": "summary",
            },
            "content": report_txt.read_text(errors="replace"),
        })

    return docs


def write_corpus(docs: list[dict], output_path: Path) -> None:
    """Write documents to a JSONL file."""
    with output_path.open("w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")


def print_summary(docs: list[dict]) -> None:
    """Print a summary of the assembled corpus."""
    by_variant: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    total_bytes = 0

    for doc in docs:
        by_variant[doc["metadata"]["design_variant"]] += 1
        by_type[doc["metadata"]["report_type"]] += 1
        total_bytes += len(doc["content"])

    print(f"\nCorpus assembled: {len(docs)} documents, {total_bytes / 1024 / 1024:.1f} MB total")
    print("\nBy design variant:")
    for variant, count in sorted(by_variant.items()):
        print(f"  {variant}: {count}")
    print("\nBy report type:")
    for rtype, count in sorted(by_type.items()):
        print(f"  {rtype}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble ASIC design corpus for RAG")
    parser.add_argument("build_dir", type=Path, help="Path to mflowgen build directory")
    parser.add_argument("-o", "--output", type=Path, default=Path("corpus.jsonl"),
                        help="Output JSONL file (default: corpus.jsonl)")
    args = parser.parse_args()

    if not args.build_dir.is_dir():
        print(f"Error: {args.build_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    docs = collect_documents(args.build_dir)
    write_corpus(docs, args.output)
    print_summary(docs)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
