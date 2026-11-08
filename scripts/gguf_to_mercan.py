#!/usr/bin/env python3
"""Wrap an upstream llama.cpp GGUF v3 model as a Mercan v1 `.mercan` file.

Mercan v1 uses GGUF v3 as its physical container, so a model whose architecture
llama.cpp already implements (e.g. LiquidAI LFM3) does not need its tensors
re-exported. This tool copies the source file's metadata and tensor data
byte-for-byte and only appends the Mercan contract keys:

    mercan.format / format_version / runtime_abi / model_family
    mercan.tokenizer.type = gguf-vocab   (llama.cpp tokenizer over tokenizer.ggml.*)
    provenance (source file name + SHA-256)

Tensor offsets in GGUF are relative to the start of the data section, so they
stay valid; only the padding between the (grown) header and the data section is
recomputed. The architecture must have a registered Mercan provider at runtime
(see `mercan arch list`).

Usage:
    gguf_to_mercan.py LFM3-1.7B-Q4_K_M.gguf --output model.mercan --model-family LFM3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import BinaryIO, Dict, List, Tuple

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# GGUF metadata value types
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = range(13)
SCALAR_SIZE = {
    T_UINT8: 1, T_INT8: 1, T_UINT16: 2, T_INT16: 2, T_UINT32: 4, T_INT32: 4,
    T_FLOAT32: 4, T_BOOL: 1, T_UINT64: 8, T_INT64: 8, T_FLOAT64: 8,
}

# Architectures with a built-in Mercan provider that delegates to llama.cpp.
SUPPORTED_ARCHITECTURES = {"lfm3"}


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError("unexpected end of GGUF file")
    return data


def read_u32(f: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(f, 4))[0]


def read_u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(f, 8))[0]


def read_string_raw(f: BinaryIO) -> bytes:
    n = read_u64(f)
    return struct.pack("<Q", n) + read_exact(f, n)


def read_value_raw(f: BinaryIO, typ: int) -> bytes:
    """Return the raw on-disk bytes of one metadata value (without its type tag)."""
    if typ in SCALAR_SIZE:
        return read_exact(f, SCALAR_SIZE[typ])
    if typ == T_STRING:
        return read_string_raw(f)
    if typ == T_ARRAY:
        elem_type = read_u32(f)
        count = read_u64(f)
        out = bytearray(struct.pack("<IQ", elem_type, count))
        if elem_type in SCALAR_SIZE:
            out += read_exact(f, SCALAR_SIZE[elem_type] * count)
        else:
            for _ in range(count):
                out += read_value_raw(f, elem_type)
        return bytes(out)
    raise ValueError(f"unsupported GGUF metadata type {typ}")


def decode_scalar(typ: int, raw: bytes):
    if typ == T_STRING:
        n = struct.unpack_from("<Q", raw)[0]
        return raw[8:8 + n].decode("utf-8")
    fmt = {T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h", T_UINT32: "<I", T_INT32: "<i",
           T_FLOAT32: "<f", T_BOOL: "<?", T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d"}.get(typ)
    return struct.unpack(fmt, raw)[0] if fmt else None


def pack_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def pack_kv(key: str, typ: int, value) -> bytes:
    out = bytearray(pack_string(key))
    out += struct.pack("<I", typ)
    if typ == T_STRING:
        out += pack_string(value)
    elif typ == T_UINT32:
        out += struct.pack("<I", int(value))
    else:
        raise ValueError(f"unsupported type for new key {key}: {typ}")
    return bytes(out)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_header(f: BinaryIO):
    if read_exact(f, 4) != GGUF_MAGIC:
        raise ValueError("not a GGUF file (bad magic)")
    version = read_u32(f)
    if version != GGUF_VERSION:
        raise ValueError(f"GGUF version {version} is not supported (Mercan v1 requires GGUF v3)")
    n_tensors = read_u64(f)
    n_kv = read_u64(f)

    kvs: List[Tuple[str, int, bytes]] = []
    for _ in range(n_kv):
        key_raw = read_string_raw(f)
        key = key_raw[8:].decode("utf-8")
        typ = read_u32(f)
        kvs.append((key, typ, read_value_raw(f, typ)))

    tensor_infos = bytearray()
    for _ in range(n_tensors):
        tensor_infos += read_string_raw(f)
        n_dims = read_u32(f)
        tensor_infos += struct.pack("<I", n_dims)
        tensor_infos += read_exact(f, 8 * n_dims)  # dims
        tensor_infos += read_exact(f, 4)           # ggml type
        tensor_infos += read_exact(f, 8)           # offset (relative to data section)
    return n_tensors, kvs, bytes(tensor_infos), f.tell()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="upstream GGUF v3 model file")
    ap.add_argument("--output", type=Path, required=True, help="output .mercan path")
    ap.add_argument("--model-family", required=True, help="value for mercan.model_family (e.g. LFM3)")
    ap.add_argument("--name", help="override general.name")
    args = ap.parse_args()

    with args.source.open("rb") as f:
        n_tensors, kvs, tensor_infos, header_end = parse_header(f)

    meta: Dict[str, Tuple[int, bytes]] = {k: (t, raw) for k, t, raw in kvs}
    if any(k.startswith("mercan.") for k in meta):
        raise SystemExit("source already contains mercan.* metadata; refusing to wrap twice")
    if "general.architecture" not in meta or meta["general.architecture"][0] != T_STRING:
        raise SystemExit("source is missing general.architecture")
    arch = decode_scalar(T_STRING, meta["general.architecture"][1])
    if arch not in SUPPORTED_ARCHITECTURES:
        raise SystemExit(f"architecture '{arch}' has no built-in Mercan provider (supported: {sorted(SUPPORTED_ARCHITECTURES)})")
    if "tokenizer.ggml.model" not in meta or "tokenizer.ggml.tokens" not in meta:
        raise SystemExit("source has no embedded tokenizer.ggml.* vocabulary")

    alignment = DEFAULT_ALIGNMENT
    if "general.alignment" in meta:
        typ, raw = meta["general.alignment"]
        alignment = int(decode_scalar(typ, raw))
    data_offset_src = align_up(header_end, alignment)

    print(f"SOURCE {args.source} arch={arch} tensors={n_tensors} kv={len(kvs)} alignment={alignment}", flush=True)
    source_sha = sha256_file(args.source)

    if args.name:
        kvs = [(k, t, pack_string(args.name)) if k == "general.name" else (k, t, raw) for k, t, raw in kvs]
        if "general.name" not in meta:
            kvs.append(("general.name", T_STRING, pack_string(args.name)))

    added = [
        ("mercan.format", T_STRING, "mercan"),
        ("mercan.format_version", T_UINT32, 1),
        ("mercan.runtime_abi", T_UINT32, 1),
        ("mercan.model_family", T_STRING, args.model_family),
        ("mercan.tokenizer.type", T_STRING, "gguf-vocab"),
        ("mercan.source_kind", T_STRING, "gguf"),
        ("mercan.source_checkpoint", T_STRING, args.source.name),
        ("mercan.source_sha256", T_STRING, source_sha),
    ]

    header = bytearray(GGUF_MAGIC)
    header += struct.pack("<IQQ", GGUF_VERSION, n_tensors, len(kvs) + len(added))
    for key, typ, raw in kvs:
        header += pack_string(key) + struct.pack("<I", typ) + raw
    for key, typ, value in added:
        header += pack_kv(key, typ, value)
    header += tensor_infos
    data_offset_dst = align_up(len(header), alignment)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with args.source.open("rb") as src, tmp.open("wb") as dst:
        dst.write(header)
        dst.write(b"\0" * (data_offset_dst - len(header)))
        src.seek(data_offset_src)
        copied = 0
        for chunk in iter(lambda: src.read(16 << 20), b""):
            dst.write(chunk)
            copied += len(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(tmp, args.output)

    expected = (args.source.stat().st_size - data_offset_src) + data_offset_dst
    if args.output.stat().st_size != expected:
        raise SystemExit(f"size check failed: {args.output.stat().st_size} != {expected}")

    with args.output.open("rb") as f:
        n_t, kvs_out, _, _ = parse_header(f)
    if n_t != n_tensors or not any(k == "mercan.format" for k, _, _ in kvs_out):
        raise SystemExit("post-write header validation failed")

    result = {
        "format": "mercan",
        "format_version": 1,
        "physical_container": "GGUF-v3",
        "architecture": arch,
        "model_family": args.model_family,
        "tokenizer_type": "gguf-vocab",
        "tensor_count": n_tensors,
        "metadata_count": len(kvs_out),
        "tensor_bytes": copied,
        "source": args.source.name,
        "source_sha256": source_sha,
        "size_bytes": args.output.stat().st_size,
        "sha256": sha256_file(args.output),
    }
    manifest = args.output.with_suffix(args.output.suffix + ".json")
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"MERCAN_WRAP_OK {args.output}", flush=True)


if __name__ == "__main__":
    main()
