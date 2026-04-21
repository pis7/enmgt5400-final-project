"""Structure-aware chunking for ASIC design artifacts.

Instead of fixed-size character splits, this chunker recognizes document
structure (section headers, report boundaries, benchmark blocks) and splits
at semantically meaningful boundaries. Each chunk gets a descriptive prefix
with metadata to improve embedding quality.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from asic_rag.config import CHUNK_SIZE, CHUNK_OVERLAP, CORPUS_PATH


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    metadata: dict


# ---------------------------------------------------------------------------
# Chunking strategies by report type
# ---------------------------------------------------------------------------

def _prefix(meta: dict) -> str:
    """Generate a descriptive prefix for a chunk to improve embedding relevance."""
    variant = meta.get("design_variant", "")
    stage = meta.get("flow_stage", "")
    rtype = meta.get("report_type", "")
    return f"[Design: {variant} | Stage: {stage} | Type: {rtype}]\n"


def _fallback_fixed_chunks(doc_id: str, text: str, meta: dict,
                           chunk_size: int, overlap: int) -> list[Chunk]:
    """Fixed-size chunking with metadata prefix — used as fallback."""
    prefix = _prefix(meta)
    chunks = []
    step = chunk_size - overlap
    start = 0
    i = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = prefix + text[start:end]
        chunks.append(Chunk(
            chunk_id=f"{doc_id}::chunk_{i}",
            text=chunk_text,
            doc_id=doc_id,
            metadata=meta,
        ))
        start += step
        i += 1
    return chunks


def _split_summary_header(doc_id: str, header: str, meta: dict, prefix: str) -> list[Chunk]:
    """Split the summary log header into semantic sub-groups for better retrieval."""
    chunks = []
    lines = header.split('\n')

    # Group lines by category
    groups: dict[str, list[str]] = {
        "design_config": [],    # timestamp, design_name, clock_period, resolved_params
        "verification": [],     # simulation pass/fail, post-run checks
        "synthesis": [],        # synth_* metrics
        "pnr": [],             # pnr_* metrics
        "sta": [],             # sta_* metrics
        "drc": [],             # drc/lvs results
    }

    in_params = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if any(k in stripped for k in ['timestamp', 'design_name', 'clock_period']):
            groups["design_config"].append(line)
        elif 'resolved_params' in stripped or in_params:
            groups["design_config"].append(line)
            in_params = stripped.startswith('- p_') or 'resolved_params' in stripped
        elif any(k in stripped for k in ['synth_setup_slack', 'synth_num_stdcells', 'synth_area', 'synthesis post-run']):
            groups["synthesis"].append(line)
        elif any(k in stripped for k in ['pnr_', 'pnr post-run']):
            groups["pnr"].append(line)
        elif any(k in stripped for k in ['sta_', 'sta post-run']):
            groups["sta"].append(line)
        elif any(k in stripped for k in ['drc', 'lvs']):
            groups["drc"].append(line)
        elif any(k in stripped for k in ['passed', 'failed', 'pickle', 'rtlsim', 'ffglsim', 'baglsim', 'pwr post-run']):
            groups["verification"].append(line)
        else:
            groups["design_config"].append(line)

    labels = {
        "design_config": "Design configuration and parameters",
        "verification": "Simulation and verification results",
        "synthesis": "Synthesis results (area, timing, cell count)",
        "pnr": "Place-and-route results (area, timing, density)",
        "sta": "Static timing analysis results (setup/hold slack)",
        "drc": "DRC and LVS verification results",
    }

    for key, group_lines in groups.items():
        if group_lines:
            text = prefix + f"{labels[key]}:\n" + '\n'.join(group_lines)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::header_{key}",
                text=text,
                doc_id=doc_id,
                metadata=meta,
            ))

    return chunks


def chunk_summary_log(doc: dict) -> list[Chunk]:
    """Chunk the 11-summarize-results/run.log by logical sections.

    Sections: header/params, flow results, and each per-benchmark block.
    """
    doc_id = doc["id"]
    meta = doc["metadata"]
    content = doc["content"]
    prefix = _prefix(meta)
    chunks = []

    # Split into benchmark blocks: each starts with " BlimpV11_sim-" or similar
    # First, separate header (everything before the first benchmark)
    bench_pattern = re.compile(r'\n (\w+_sim-\w[\w-]*)\n')
    bench_matches = list(bench_pattern.finditer(content))

    if not bench_matches:
        # No benchmarks found, chunk as single doc
        return [Chunk(chunk_id=f"{doc_id}::chunk_0", text=prefix + content,
                      doc_id=doc_id, metadata=meta)]

    # Header section: split into semantic sub-groups instead of one big chunk
    header_end = bench_matches[0].start()
    header_text = content[:header_end].strip()
    if header_text:
        chunks.extend(_split_summary_header(doc_id, header_text, meta, prefix))

    # Each benchmark block
    for idx, match in enumerate(bench_matches):
        bench_name = match.group(1)
        start = match.start()
        end = bench_matches[idx + 1].start() if idx + 1 < len(bench_matches) else len(content)
        block = content[start:end].strip()
        if block:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::bench_{bench_name}",
                text=prefix + f"Benchmark results for {bench_name}:\n" + block,
                doc_id=doc_id,
                metadata=meta,
            ))

    return chunks


def chunk_batch_report(doc: dict) -> list[Chunk]:
    """Chunk report.txt by its major sections (DESIGN PARAMETERS, AREA & TIMING, etc).

    Each section is a self-contained comparison table across all design variants.
    """
    doc_id = doc["id"]
    meta = doc["metadata"]
    content = doc["content"]
    prefix = _prefix(meta)
    chunks = []

    # Sections are delimited by lines like " ENERGY (nJ)" or " AREA & TIMING"
    section_pattern = re.compile(r'\n( [A-Z][A-Z &]+(?:\([^)]*\))?)\n ----')
    section_matches = list(section_pattern.finditer(content))

    if not section_matches:
        return [Chunk(chunk_id=f"{doc_id}::chunk_0", text=prefix + content,
                      doc_id=doc_id, metadata=meta)]

    # Preamble (header before first section)
    preamble_end = section_matches[0].start()
    preamble = content[:preamble_end].strip()
    if preamble:
        chunks.append(Chunk(
            chunk_id=f"{doc_id}::header",
            text=prefix + preamble,
            doc_id=doc_id,
            metadata=meta,
        ))

    # Each section
    for idx, match in enumerate(section_matches):
        section_name = match.group(1).strip()
        start = match.start()
        end = section_matches[idx + 1].start() if idx + 1 < len(section_matches) else len(content)
        section_text = content[start:end].strip()

        # Large sections (e.g. ENERGY with 28 benchmarks) might still be big
        # but they're logically coherent — keep them whole if under 2x chunk_size,
        # otherwise split by row groups
        if len(section_text) <= CHUNK_SIZE * 3:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::section_{section_name.lower().replace(' ', '_').replace('&', 'and')}",
                text=prefix + f"Cross-variant comparison — {section_name}:\n" + section_text,
                doc_id=doc_id,
                metadata=meta,
            ))
        else:
            # Split large sections into sub-chunks with header context
            lines = section_text.split('\n')
            # Find the header lines (first few lines with column headers + separator)
            header_lines = []
            data_lines = []
            for line in lines:
                if not data_lines and (line.strip().startswith('-') or not line.strip()
                                       or any(c.isalpha() for c in line[:30])):
                    header_lines.append(line)
                else:
                    data_lines.append(line)

            table_header = '\n'.join(header_lines)
            # Group data lines into sub-chunks
            sub_chunk_lines = []
            sub_idx = 0
            for line in data_lines:
                sub_chunk_lines.append(line)
                if len('\n'.join(sub_chunk_lines)) > CHUNK_SIZE:
                    chunk_text = (prefix + f"Cross-variant comparison — {section_name}:\n"
                                  + table_header + '\n' + '\n'.join(sub_chunk_lines))
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}::section_{section_name.lower().replace(' ', '_')}_{sub_idx}",
                        text=chunk_text,
                        doc_id=doc_id,
                        metadata=meta,
                    ))
                    sub_chunk_lines = []
                    sub_idx += 1

            if sub_chunk_lines:
                chunk_text = (prefix + f"Cross-variant comparison — {section_name}:\n"
                              + table_header + '\n' + '\n'.join(sub_chunk_lines))
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}::section_{section_name.lower().replace(' ', '_')}_{sub_idx}",
                    text=chunk_text,
                    doc_id=doc_id,
                    metadata=meta,
                ))

    return chunks


def chunk_power_report(doc: dict) -> list[Chunk]:
    """Chunk power reports as single documents with descriptive prefix."""
    doc_id = doc["id"]
    meta = doc["metadata"]
    prefix = _prefix(meta)

    # Extract benchmark name from doc_id (e.g. "BlimpV11_sim-vvadd-summary.rpt")
    bench_match = re.search(r'sim-(\w[\w-]*?)-(summary|detailed)', doc_id)
    bench_name = bench_match.group(1) if bench_match else "unknown"
    detail_level = bench_match.group(2) if bench_match else "unknown"

    text = (prefix + f"Power analysis ({detail_level}) for benchmark {bench_name}:\n"
            + doc["content"])

    # Summary reports are small enough for a single chunk
    if len(text) <= CHUNK_SIZE * 2:
        return [Chunk(chunk_id=f"{doc_id}::chunk_0", text=text,
                      doc_id=doc_id, metadata=meta)]

    # Detailed reports need splitting
    return _fallback_fixed_chunks(doc_id, doc["content"], meta, CHUNK_SIZE, CHUNK_OVERLAP)


def chunk_timing_report(doc: dict) -> list[Chunk]:
    """Chunk timing reports by splitting at path boundaries.

    Timing reports contain multiple paths, each starting with 'Startpoint:'.
    """
    doc_id = doc["id"]
    meta = doc["metadata"]
    content = doc["content"]
    prefix = _prefix(meta)

    # Split on "Startpoint:" which begins each timing path
    path_splits = re.split(r'(?=\s+Startpoint:)', content)

    if len(path_splits) <= 1:
        # No path structure found (maybe a summary report)
        text = prefix + content
        if len(text) <= CHUNK_SIZE * 2:
            return [Chunk(chunk_id=f"{doc_id}::chunk_0", text=text,
                          doc_id=doc_id, metadata=meta)]
        return _fallback_fixed_chunks(doc_id, content, meta, CHUNK_SIZE, CHUNK_OVERLAP)

    chunks = []
    # First part is the report header
    header = path_splits[0].strip()
    if header:
        chunks.append(Chunk(
            chunk_id=f"{doc_id}::header",
            text=prefix + "Timing report header:\n" + header,
            doc_id=doc_id,
            metadata=meta,
        ))

    # Each timing path as a chunk (or group small paths together)
    path_group = []
    path_group_len = 0
    group_idx = 0

    for path_text in path_splits[1:]:
        path_text = path_text.strip()
        if not path_text:
            continue

        # If a single path exceeds chunk_size, split it with fallback
        if len(path_text) > CHUNK_SIZE:
            # Flush current group first
            if path_group:
                combined = prefix + "Timing paths:\n" + "\n\n".join(path_group)
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}::paths_{group_idx}",
                    text=combined,
                    doc_id=doc_id,
                    metadata=meta,
                ))
                path_group = []
                path_group_len = 0
                group_idx += 1
            # Split the large path
            chunks.extend(_fallback_fixed_chunks(
                f"{doc_id}::paths_{group_idx}", path_text, meta, CHUNK_SIZE, CHUNK_OVERLAP))
            group_idx += 1
            continue

        if path_group_len + len(path_text) > CHUNK_SIZE and path_group:
            combined = prefix + "Timing paths:\n" + "\n\n".join(path_group)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::paths_{group_idx}",
                text=combined,
                doc_id=doc_id,
                metadata=meta,
            ))
            path_group = []
            path_group_len = 0
            group_idx += 1

        path_group.append(path_text)
        path_group_len += len(path_text)

    if path_group:
        combined = prefix + "Timing paths:\n" + "\n\n".join(path_group)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}::paths_{group_idx}",
            text=combined,
            doc_id=doc_id,
            metadata=meta,
        ))

    return chunks


def chunk_area_report(doc: dict) -> list[Chunk]:
    """Chunk area reports — keep header + summary, split cell table if needed."""
    doc_id = doc["id"]
    meta = doc["metadata"]
    content = doc["content"]
    prefix = _prefix(meta)

    text = prefix + "Area report:\n" + content
    if len(text) <= CHUNK_SIZE * 2:
        return [Chunk(chunk_id=f"{doc_id}::chunk_0", text=text,
                      doc_id=doc_id, metadata=meta)]

    return _fallback_fixed_chunks(doc_id, content, meta, CHUNK_SIZE, CHUNK_OVERLAP)


def chunk_drc_report(doc: dict) -> list[Chunk]:
    """Chunk DRC reports — split if large."""
    doc_id = doc["id"]
    meta = doc["metadata"]
    prefix = _prefix(meta)
    text = prefix + "DRC report:\n" + doc["content"]
    if len(text) <= CHUNK_SIZE * 2:
        return [Chunk(chunk_id=f"{doc_id}::chunk_0", text=text,
                      doc_id=doc_id, metadata=meta)]
    return _fallback_fixed_chunks(doc_id, doc["content"], meta, CHUNK_SIZE, CHUNK_OVERLAP)


def chunk_build_config(doc: dict) -> list[Chunk]:
    """Chunk build config YAMLs — small, keep whole."""
    doc_id = doc["id"]
    meta = doc["metadata"]
    prefix = _prefix(meta)
    text = prefix + "Build configuration:\n" + doc["content"]
    return [Chunk(chunk_id=f"{doc_id}::chunk_0", text=text,
                  doc_id=doc_id, metadata=meta)]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def chunk_document(doc: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Route a document to the appropriate structure-aware chunker."""
    report_type = doc["metadata"].get("report_type", "")
    flow_stage = doc["metadata"].get("flow_stage", "")

    # Summary log from summarize-results
    if flow_stage == "11-summarize-results" and report_type == "summary":
        return chunk_summary_log(doc)

    # Cross-variant batch report
    if report_type == "summary" and doc["id"].endswith("report.txt"):
        return chunk_batch_report(doc)

    # Power reports
    if report_type == "power-report":
        return chunk_power_report(doc)

    # Timing reports
    if report_type == "timing-report":
        return chunk_timing_report(doc)

    # Area reports
    if report_type == "area-report":
        return chunk_area_report(doc)

    # DRC reports
    if report_type == "drc-report":
        return chunk_drc_report(doc)

    # Build config
    if report_type == "build-config":
        return chunk_build_config(doc)

    # Everything else: fixed-size with prefix
    return _fallback_fixed_chunks(doc["id"], doc["content"], doc["metadata"],
                                  chunk_size, overlap)


def load_and_chunk_corpus(corpus_path: Path = CORPUS_PATH) -> list[Chunk]:
    """Load corpus.jsonl and return all chunks using structure-aware chunking."""
    chunks = []
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            chunks.extend(chunk_document(doc))
    return chunks
