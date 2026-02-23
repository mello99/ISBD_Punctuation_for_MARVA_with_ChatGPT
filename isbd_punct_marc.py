#!/usr/bin/env python3
"""
Created with ChatGPT 5.2

Apply conservative ISBD-style punctuation normalization to selected MARC fields:
1XX/245/250/264/300/490/6XX/7XX

What it does:
- 245: ensure ": $b" and "/ $c" patterns (and keeps $n/$p after $a)
- 250: comma between $a and $b when both present; normalize trailing punctuation
- 264 (ind2=1): "Place : Publisher, Date" pattern using $a/$b/$c
- 300: "$a : $b ; $c + $e"
- 490: "$a ; $v"
- 1XX/6XX/7XX: ONLY normalize terminal punctuation (do not inject ISBD separators)

Usage:
  python3 isbd_punct_marc.py --in input.mrc --out output_isbd.mrc

Optional: also output a CSV showing before/after strings for each edited field:
  python3 isbd_punct_marc.py --in input.mrc --out output_isbd.mrc --audit audit.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from pymarc import MARCReader, MARCWriter, Record, Field


# ----------------------------
# Helpers
# ----------------------------

TRAILING_SPACE_PUNCT = " \t\r\n"
SOFT_TRAILING_PUNCT = " /:;=,.-"
HARD_ENDING = (".", ")", "]", "?", "!", "…")  # do not add period after these


def clean(s: Optional[str]) -> str:
    if not s:
        return ""
    return " ".join(s.split()).strip()


def rstrip_spaces(s: str) -> str:
    return s.rstrip(TRAILING_SPACE_PUNCT)


def rstrip_soft_punct(s: str) -> str:
    return s.rstrip(SOFT_TRAILING_PUNCT).rstrip(TRAILING_SPACE_PUNCT)


def ensure_terminal_period(s: str) -> str:
    s = rstrip_spaces(s)
    if not s:
        return ""
    if s.endswith(HARD_ENDING):
        return s
    # don't double-dot if already ends with dot
    if s.endswith("."):
        return s
    return s + "."


def iter_subfield_pairs(field: Field) -> List[Tuple[str, str]]:
    """Return list of (code, value) pairs in order."""
    if not getattr(field, "subfields", None):
        return []
    sf = field.subfields
    return list(zip(sf[0::2], sf[1::2]))


def rebuild_field(tag: str, ind1: str, ind2: str, pairs: List[Tuple[str, str]]) -> Field:
    subfields: List[str] = []
    for code, val in pairs:
        subfields.extend([code, val])
    return Field(tag=tag, indicators=[ind1, ind2], subfields=subfields)


def field_to_str(field: Field) -> str:
    """Stringify a field's subfields for auditing."""
    parts = []
    for code, val in iter_subfield_pairs(field):
        parts.append(f"${code} {val}")
    return " ".join(parts)


# ----------------------------
# Field-specific ISBD-ish logic
# ----------------------------

def punct_245(field: Field) -> Field:
    pairs = iter_subfield_pairs(field)
    if not pairs:
        return field

    # Extract common subfields while preserving unknown ones (conservative).
    # We'll rebuild a/p/n/b/c in a normalized way, then append other subfields as-is.
    a_vals, n_vals, p_vals, b_vals, c_vals = [], [], [], [], []
    other: List[Tuple[str, str]] = []

    for code, val in pairs:
        v = clean(val)
        if not v:
            continue
        if code == "a":
            a_vals.append(v)
        elif code == "n":
            n_vals.append(v)
        elif code == "p":
            p_vals.append(v)
        elif code == "b":
            b_vals.append(v)
        elif code == "c":
            c_vals.append(v)
        else:
            other.append((code, v))

    # Build $a including $n/$p after it (common display practice)
    # We strip trailing punctuation from a/n/p to avoid doubles.
    a_text = " ".join([rstrip_soft_punct(x) for x in (a_vals + n_vals + p_vals) if x]).strip()

    b_text = " ".join([rstrip_soft_punct(x) for x in b_vals if x]).strip()
    c_text = " ".join([rstrip_soft_punct(x) for x in c_vals if x]).strip()

    new_pairs: List[Tuple[str, str]] = []
    if a_text:
        # Do not force a terminal period inside 245; we manage separators.
        new_pairs.append(("a", a_text))

    if b_text:
        # Ensure $a ends cleanly; add $b as-is (MARC expects punctuation in data, but we keep value clean)
        new_pairs.append(("b", b_text))

    if c_text:
        new_pairs.append(("c", c_text))

    # Append other subfields unchanged (but cleaned)
    new_pairs.extend(other)

    # Now inject punctuation by editing the *values* where appropriate:
    # MARC practice often stores punctuation at end of $a when followed by $b, and at end of $b when followed by $c.
    # We'll apply:
    #   $a ends with " :" if $b exists
    #   $b ends with " /" if $c exists
    #   $a ends with " /" if no $b but $c exists
    # Additionally, keep '=' out unless user wants; many records handle parallel titles with $b anyway.

    # Find indexes
    idx_a = next((i for i, (c, _) in enumerate(new_pairs) if c == "a"), None)
    idx_b = next((i for i, (c, _) in enumerate(new_pairs) if c == "b"), None)
    idx_c = next((i for i, (c, _) in enumerate(new_pairs) if c == "c"), None)

    # Helper to set suffix
    def add_suffix(i: int, suffix: str) -> None:
        code, val = new_pairs[i]
        val = rstrip_soft_punct(val)
        new_pairs[i] = (code, val + suffix)

    if idx_a is not None and idx_b is not None:
        add_suffix(idx_a, " :")
    if idx_b is not None and idx_c is not None:
        add_suffix(idx_b, " /")
    if idx_a is not None and idx_b is None and idx_c is not None:
        add_suffix(idx_a, " /")

    # Ensure final subfield ends with period (common in 245)
    # BUT some institutions avoid forcing period; you asked to apply ISBD punctuation,
    # so we add a terminal period if missing.
    # Determine last textual subfield among a/b/c/others
    last_i = None
    for i in range(len(new_pairs) - 1, -1, -1):
        if new_pairs[i][0].isalnum():
            last_i = i
            break
    if last_i is not None:
        code, val = new_pairs[last_i]
        new_pairs[last_i] = (code, ensure_terminal_period(val))

    return rebuild_field("245", field.indicator1, field.indicator2, new_pairs)


def punct_250(field: Field) -> Field:
    pairs = [(c, clean(v)) for c, v in iter_subfield_pairs(field) if clean(v)]
    if not pairs:
        return field

    # If both $a and $b exist, ensure $a ends with comma.
    idx_a = next((i for i, (c, _) in enumerate(pairs) if c == "a"), None)
    idx_b = next((i for i, (c, _) in enumerate(pairs) if c == "b"), None)

    if idx_a is not None and idx_b is not None:
        ca, va = pairs[idx_a]
        pairs[idx_a] = (ca, rstrip_soft_punct(va) + ",")

    # Ensure final period
    pairs[-1] = (pairs[-1][0], ensure_terminal_period(pairs[-1][1]))
    return rebuild_field("250", field.indicator1, field.indicator2, pairs)


def punct_264_pub(field: Field) -> Field:
    """
    Apply ISBD-ish punctuation for publication statements (264 ind2=1):
      $a ends with " :" if $b exists, else "," if $c exists
      $b ends with "," if $c exists
      ensure final period
    Multiple $a/$b/$c are left in place; we punctuate only the last occurrence
    of each "area" conservatively.
    """
    pairs = [(c, clean(v)) for c, v in iter_subfield_pairs(field) if clean(v)]
    if not pairs:
        return field

    # Identify last indexes of a/b/c
    idx_a = max([i for i, (c, _) in enumerate(pairs) if c == "a"], default=None)
    idx_b = max([i for i, (c, _) in enumerate(pairs) if c == "b"], default=None)
    idx_c = max([i for i, (c, _) in enumerate(pairs) if c == "c"], default=None)

    def set_suffix(i: int, suffix: str) -> None:
        code, val = pairs[i]
        pairs[i] = (code, rstrip_soft_punct(val) + suffix)

    if idx_a is not None:
        if idx_b is not None:
            set_suffix(idx_a, " :")
        elif idx_c is not None:
            set_suffix(idx_a, ",")
    if idx_b is not None and idx_c is not None:
        set_suffix(idx_b, ",")

    # Final period
    pairs[-1] = (pairs[-1][0], ensure_terminal_period(pairs[-1][1]))
    return rebuild_field("264", field.indicator1, field.indicator2, pairs)


def punct_300(field: Field) -> Field:
    pairs = [(c, clean(v)) for c, v in iter_subfield_pairs(field) if clean(v)]
    if not pairs:
        return field

    idx_a = next((i for i, (c, _) in enumerate(pairs) if c == "a"), None)
    idx_b = next((i for i, (c, _) in enumerate(pairs) if c == "b"), None)
    idx_c = next((i for i, (c, _) in enumerate(pairs) if c == "c"), None)
    idx_e = next((i for i, (c, _) in enumerate(pairs) if c == "e"), None)

    def set_suffix(i: int, suffix: str) -> None:
        code, val = pairs[i]
        pairs[i] = (code, rstrip_soft_punct(val) + suffix)

    if idx_a is not None and idx_b is not None:
        set_suffix(idx_a, " :")
    if idx_b is not None and idx_c is not None:
        set_suffix(idx_b, " ;")
    if idx_c is not None and idx_e is not None:
        set_suffix(idx_c, " +")

    # Final period (common)
    pairs[-1] = (pairs[-1][0], ensure_terminal_period(pairs[-1][1]))
    return rebuild_field("300", field.indicator1, field.indicator2, pairs)


def punct_490(field: Field) -> Field:
    pairs = [(c, clean(v)) for c, v in iter_subfield_pairs(field) if clean(v)]
    if not pairs:
        return field

    # If $a and $v exist, ensure $a ends with ';'
    idx_a = next((i for i, (c, _) in enumerate(pairs) if c == "a"), None)
    idx_v = next((i for i, (c, _) in enumerate(pairs) if c == "v"), None)

    if idx_a is not None and idx_v is not None:
        ca, va = pairs[idx_a]
        pairs[idx_a] = (ca, rstrip_soft_punct(va) + " ;")

    pairs[-1] = (pairs[-1][0], ensure_terminal_period(pairs[-1][1]))
    return rebuild_field("490", field.indicator1, field.indicator2, pairs)


def punct_heading_terminal(field: Field) -> Field:
    """
    For 1XX/6XX/7XX: do NOT inject ISBD separators.
    Only normalize terminal punctuation on the final subfield.
    """
    if field.is_control_field():
        return field

    pairs = [(c, clean(v)) for c, v in iter_subfield_pairs(field) if clean(v)]
    if not pairs:
        return field

    # Some headings legitimately end with '-' (open-ended) or ')' etc.; we avoid forcing period in those cases.
    code_last, val_last = pairs[-1]
    pairs[-1] = (code_last, ensure_terminal_period(val_last))
    return rebuild_field(field.tag, field.indicator1, field.indicator2, pairs)


# ----------------------------
# Apply to record
# ----------------------------

def apply_isbd(record: Record) -> Tuple[Record, List[Tuple[str, str, str]]]:
    """
    Returns (modified_record, audit_rows)
    audit_rows: list of (tag, before, after) for changed fields
    """
    audit: List[Tuple[str, str, str]] = []
    new_fields: List[Field] = []

    for f in record.fields:
        if isinstance(f, Field) and not f.is_control_field():
            before = field_to_str(f)

            if f.tag == "245":
                nf = punct_245(f)
            elif f.tag == "250":
                nf = punct_250(f)
            elif f.tag == "264" and f.indicator2 == "1":
                nf = punct_264_pub(f)
            elif f.tag == "300":
                nf = punct_300(f)
            elif f.tag == "490":
                nf = punct_490(f)
            elif f.tag.startswith("1") and len(f.tag) == 3 and f.tag[0] == "1":
                nf = punct_heading_terminal(f)
            elif f.tag.startswith("6") and len(f.tag) == 3 and f.tag[0] == "6":
                nf = punct_heading_terminal(f)
            elif f.tag.startswith("7") and len(f.tag) == 3 and f.tag[0] == "7":
                nf = punct_heading_terminal(f)
            else:
                nf = f

            after = field_to_str(nf)
            if after != before:
                audit.append((f.tag, before, after))
            new_fields.append(nf)
        else:
            # control fields and anything else unchanged
            new_fields.append(f)

    record.fields = new_fields
    return record, audit


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply conservative ISBD punctuation normalization to MARC records.")
    p.add_argument("--in", dest="in_path", required=True, help="Input .mrc file")
    p.add_argument("--out", dest="out_path", required=True, help="Output .mrc file")
    p.add_argument("--audit", dest="audit_csv", default="", help="Optional CSV audit output (tag,before,after)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    audit_path = Path(args.audit_csv) if args.audit_csv else None

    if not in_path.exists():
        print(f"ERROR: Input not found: {in_path}", file=sys.stderr)
        return 2

    audit_rows_all: List[Tuple[str, str, str, str]] = []  # (001, tag, before, after)

    with in_path.open("rb") as fh, out_path.open("wb") as out_fh:
        reader = MARCReader(fh, to_unicode=True, force_utf8=True)
        writer = MARCWriter(out_fh)

        count = 0
        for rec in reader:
            count += 1
            rec_id = (rec["001"].data.strip() if rec["001"] and rec["001"].data else f"REC_{count}")

            new_rec, audit_rows = apply_isbd(rec)
            writer.write(new_rec)

            for tag, before, after in audit_rows:
                audit_rows_all.append((rec_id, tag, before, after))

        writer.close()

    print(f"Done. Records processed: {count}")
    print(f"Output MARC: {out_path}")

    if audit_path:
        with audit_path.open("w", newline="", encoding="utf-8") as csv_fh:
            w = csv.writer(csv_fh)
            w.writerow(["001", "tag", "before", "after"])
            w.writerows(audit_rows_all)
        print(f"Audit CSV: {audit_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
