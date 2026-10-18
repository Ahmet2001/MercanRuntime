#!/usr/bin/env python3
"""Convert a Mercan/NedoLM PyTorch or HF checkpoint into a single .mercan file.

Mercan v1 intentionally uses the GGUF v3 physical container so libmercan can
reuse the proven ggml/llama tensor loader. The .mercan contract is stricter:
it embeds the exact tokenizer surface-vocabulary asset, model architecture,
MorphFFN routing tensor, chat template and provenance metadata in one file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

ALIGN = 32
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10

DEFAULT_CHECKPOINT = Path(
    "/arf/scratch/egitimg16/nedolm_0p8b/runs/"
    "nedolm-0p8b-sft-full-v1/checkpoints/step_00023334.pt"
)
DEFAULT_VOCAB_BIN = Path("/arf/scratch/egitimg16/NedoTokenizer/assets/surface-vocab.bin")
DEFAULT_VOCAB_TXT = Path("/arf/scratch/egitimg16/NedoTokenizer/assets/surface-vocab.txt")
DEFAULT_ROLE_TABLE = Path("/arf/scratch/egitimg16/nedolm_0p8b/assets/token_role_table.pt")
DEFAULT_OUTPUT = Path(
    "/arf/scratch/egitimg16/nedolm_0p8b/mercan_export/"
    "Mercan-0.8B-SFT-step-00023334-F16.mercan"
)
EXPECTED_NDSRF004_SHA256 = "72412d981dac65a29d1767bc98821fc2bcffc2de53c534e7c719598515bfb600"

DEFAULT_ARCH = {
    "vocab_size": 32000,
    "context_length": 4096,
    "embedding_length": 1536,
    "block_count": 24,
    "feed_forward_length": 5632,
    "head_count": 12,
    "head_count_kv": 4,
    "head_dim": 128,
    "rms_epsilon": 1.0e-6,
    "sliding_window": 2048,
    "rope_freq_base": 1_000_000.0,
    "morph_layer_count": 18,
    "morph_shared_width": 4224,
    "morph_root_width": 704,
    "morph_suffix_width": 704,
    "tie_word_embeddings": True,
}


def align_up(value: int, alignment: int = ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_string(value: str | bytes) -> bytes:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def pack_meta(key: str, typ: int, value) -> bytes:
    out = bytearray(pack_string(key))
    out += struct.pack("<I", typ)
    if typ == GGUF_TYPE_STRING:
        out += pack_string(value)
    elif typ == GGUF_TYPE_UINT32:
        out += struct.pack("<I", int(value))
    elif typ == GGUF_TYPE_UINT64:
        out += struct.pack("<Q", int(value))
    elif typ == GGUF_TYPE_FLOAT32:
        out += struct.pack("<f", float(value))
    elif typ == GGUF_TYPE_BOOL:
        out += struct.pack("<?", bool(value))
    elif typ == GGUF_TYPE_ARRAY:
        elem_type, values = value
        out += struct.pack("<I", elem_type)
        out += struct.pack("<Q", len(values))
        if elem_type == GGUF_TYPE_STRING:
            for item in values:
                out += pack_string(item)
        elif elem_type == GGUF_TYPE_INT32:
            out += struct.pack("<" + "i" * len(values), *[int(x) for x in values])
        elif elem_type == GGUF_TYPE_UINT8:
            out += bytes(values)
        else:
            raise ValueError(f"unsupported GGUF array type: {elem_type}")
    else:
        raise ValueError(f"unsupported GGUF metadata type: {typ}")
    return bytes(out)


def parse_vocab_txt(path: Path, vocab_size: int) -> Tuple[List[bytes], List[int]]:
    rows: List[Tuple[int, bytes, int]] = []
    for line in path.read_text("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 4:
            continue
        token_id = int(cols[0])
        token_kind = cols[1]
        hex_bytes = cols[3].strip()
        if token_kind == "special":
            special_pieces = {
                0: b"<PAD>",
                1: b"<BOS>",
                2: b"<EOS>",
                32000: b"<|im_start|>",
                32001: b"<|im_end|>",
            }
            if token_id not in special_pieces:
                raise RuntimeError(f"unsupported special token id in vocab: {token_id}")
            piece = special_pieces[token_id]
            token_type = 3  # GGML CONTROL
        else:
            piece = bytes.fromhex(hex_bytes)
            token_type = 6 if token_kind == "base-byte" else 1  # BYTE / NORMAL
        rows.append((token_id, piece, token_type))
    rows.sort(key=lambda row: row[0])
    ids = [row[0] for row in rows]
    if len(rows) != vocab_size or ids != list(range(vocab_size)):
        raise RuntimeError(f"surface-vocab.txt IDs are not exactly 0..{vocab_size - 1}")
    return [row[1] for row in rows], [row[2] for row in rows]


def strip_common_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("_orig_mod.", "module.", "model.")
    out = dict(state_dict)
    changed = True
    while changed and out:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in out):
                out = {key[len(prefix):]: value for key, value in out.items()}
                changed = True
                break
    return out


def load_hf_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    safetensor_files = sorted(path.glob("*.safetensors"))
    if safetensor_files:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("HF safetensors input requires `pip install safetensors`") from exc
        state: Dict[str, torch.Tensor] = {}
        for shard in safetensor_files:
            for key, value in load_file(str(shard), device="cpu").items():
                if key in state:
                    raise RuntimeError(f"duplicate HF tensor across shards: {key}")
                state[key] = value
        return strip_common_prefixes(state)

    for name in ("pytorch_model.bin", "model.pt", "model.bin"):
        candidate = path / name
        if candidate.is_file():
            obj = torch.load(candidate, map_location="cpu", mmap=True, weights_only=False)
            if isinstance(obj, dict) and "state_dict" in obj:
                obj = obj["state_dict"]
            if not isinstance(obj, dict):
                raise RuntimeError(f"unsupported HF weight object in {candidate}")
            return strip_common_prefixes(obj)
    raise FileNotFoundError(f"no *.safetensors or pytorch_model.bin found in {path}")


def load_state_dict(source: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    provenance: Dict[str, object] = {}
    if source.is_dir():
        state = load_hf_state_dict(source)
        provenance["source_kind"] = "huggingface"
        provenance["source_path"] = str(source)
        config_path = source / "config.json"
        if config_path.is_file():
            provenance["hf_config"] = json.loads(config_path.read_text("utf-8"))
        return state, provenance

    obj = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    provenance["source_kind"] = "pytorch"
    provenance["source_path"] = str(source)
    if isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        state = obj["model"]
        for key in ("step", "base_checkpoint", "tokens_seen", "assistant_tokens_seen"):
            if key in obj:
                provenance[key] = obj[key]
    elif isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        state = obj["state_dict"]
    elif isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        state = obj
    else:
        raise RuntimeError("unsupported PyTorch checkpoint: expected model/state_dict tensor mapping")
    return strip_common_prefixes(state), provenance


def resolve_key(state: Dict[str, torch.Tensor], candidates: Iterable[str]) -> str:
    for key in candidates:
        if key in state:
            return key
    raise KeyError(f"none of the tensor aliases exist: {list(candidates)}")


def map_tensors(
    state: Dict[str, torch.Tensor], arch: Dict[str, object], role_table_path: Path
) -> List[Tuple[str, torch.Tensor, int]]:
    mapped: List[Tuple[str, torch.Tensor, int]] = []

    def add(dst: str, *aliases: str) -> None:
        src = resolve_key(state, aliases)
        tensor = state[src]
        if not torch.is_tensor(tensor):
            raise TypeError(f"{src} is not a tensor")
        ggml_type = GGML_TYPE_F32 if tensor.ndim == 1 else GGML_TYPE_F16
        mapped.append((dst, tensor, ggml_type))

    add("token_embd.weight", "embed_tokens.weight", "tok_embeddings.weight", "model.embed_tokens.weight")
    layer_count = int(arch["block_count"])
    morph_layers = int(arch["morph_layer_count"])

    for i in range(layer_count):
        p = f"layers.{i}"
        hp = f"model.layers.{i}"
        add(f"blk.{i}.attn_norm.weight", f"{p}.attn_norm.weight", f"{hp}.attn_norm.weight", f"{hp}.input_layernorm.weight")
        add(f"blk.{i}.attn_q.weight", f"{p}.attn.q_proj.weight", f"{hp}.attn.q_proj.weight", f"{hp}.self_attn.q_proj.weight")
        add(f"blk.{i}.attn_k.weight", f"{p}.attn.k_proj.weight", f"{hp}.attn.k_proj.weight", f"{hp}.self_attn.k_proj.weight")
        add(f"blk.{i}.attn_v.weight", f"{p}.attn.v_proj.weight", f"{hp}.attn.v_proj.weight", f"{hp}.self_attn.v_proj.weight")
        add(f"blk.{i}.attn_output.weight", f"{p}.attn.o_proj.weight", f"{hp}.attn.o_proj.weight", f"{hp}.self_attn.o_proj.weight")
        add(f"blk.{i}.ffn_norm.weight", f"{p}.ffn_norm.weight", f"{hp}.ffn_norm.weight", f"{hp}.post_attention_layernorm.weight")
        if i < morph_layers:
            add(f"blk.{i}.ffn_gate.weight", f"{p}.ffn.gate_weight", f"{hp}.ffn.gate_weight")
            add(f"blk.{i}.ffn_up.weight", f"{p}.ffn.up_weight", f"{hp}.ffn.up_weight")
            add(f"blk.{i}.ffn_down.weight", f"{p}.ffn.down_weight", f"{hp}.ffn.down_weight")
        else:
            add(f"blk.{i}.ffn_gate.weight", f"{p}.ffn.gate_proj.weight", f"{hp}.ffn.gate_proj.weight", f"{hp}.mlp.gate_proj.weight")
            add(f"blk.{i}.ffn_up.weight", f"{p}.ffn.up_proj.weight", f"{hp}.ffn.up_proj.weight", f"{hp}.mlp.up_proj.weight")
            add(f"blk.{i}.ffn_down.weight", f"{p}.ffn.down_proj.weight", f"{hp}.ffn.down_proj.weight", f"{hp}.mlp.down_proj.weight")
    add("output_norm.weight", "norm.weight", "model.norm.weight")

    role_aliases = (
        "layers.0.ffn.token_roles",
        "model.layers.0.ffn.token_roles",
    )
    roles = None
    for alias in role_aliases:
        if alias in state:
            roles = state[alias]
            break
    if roles is None:
        role_obj = torch.load(role_table_path, map_location="cpu", weights_only=True)
        roles = role_obj.get("token_roles") if isinstance(role_obj, dict) else role_obj
    roles = roles.reshape(-1).to(torch.uint8)
    vocab_size = int(arch["vocab_size"])
    if roles.numel() != vocab_size:
        raise RuntimeError(f"role table has {roles.numel()} entries; expected {vocab_size}")
    role_mask = torch.stack([(roles == 1), (roles == 2)], dim=1).to(torch.float32)
    mapped.append(("blk.0.ffn_gate_tid2eid.weight", role_mask, GGML_TYPE_F32))
    return mapped


def build_metadata(
    arch: Dict[str, object],
    vocab_bytes: bytes,
    vocab_sha: str,
    tokens: List[bytes],
    token_types: List[int],
    provenance: Dict[str, object],
    model_name: str,
) -> List[Tuple[str, int, object]]:
    checkpoint_name = Path(str(provenance.get("source_path", "unknown"))).name
    base_checkpoint = str(provenance.get("base_checkpoint", "unknown"))
    metadata = [
        ("general.architecture", GGUF_TYPE_STRING, "nedolm"),
        ("general.name", GGUF_TYPE_STRING, model_name),
        ("general.description", GGUF_TYPE_STRING, "Mercan Turkish assistant model with token-prior MorphFFN"),
        ("general.file_type", GGUF_TYPE_UINT32, 1),
        ("general.alignment", GGUF_TYPE_UINT32, ALIGN),
        ("mercan.format", GGUF_TYPE_STRING, "mercan"),
        ("mercan.format_version", GGUF_TYPE_UINT32, 1),
        ("mercan.runtime_abi", GGUF_TYPE_UINT32, 1),
        ("mercan.model_family", GGUF_TYPE_STRING, "Mercan"),
        ("mercan.source_kind", GGUF_TYPE_STRING, str(provenance.get("source_kind", "unknown"))),
        ("mercan.source_checkpoint", GGUF_TYPE_STRING, checkpoint_name),
        ("mercan.base_checkpoint", GGUF_TYPE_STRING, base_checkpoint),
        ("mercan.chat_template", GGUF_TYPE_STRING, "chatml_tr"),
        ("mercan.chat_template_spec", GGUF_TYPE_STRING, "<|im_start|>{rol}\\n{content}<|im_end|>\\n; roles=sistem,kullanici,asistan; EOS once at conversation end"),
        ("mercan.tokenizer.type", GGUF_TYPE_STRING, "ndsurf004"),
        ("mercan.tokenizer.spec", GGUF_TYPE_STRING, "NDSRF004"),
        ("mercan.tokenizer.surface_vocab_sha256", GGUF_TYPE_STRING, vocab_sha),
        ("mercan.tokenizer.surface_vocab", GGUF_TYPE_ARRAY, (GGUF_TYPE_UINT8, vocab_bytes)),
        ("nedolm.vocab_size", GGUF_TYPE_UINT32, arch["vocab_size"]),
        ("nedolm.context_length", GGUF_TYPE_UINT32, arch["context_length"]),
        ("nedolm.embedding_length", GGUF_TYPE_UINT32, arch["embedding_length"]),
        ("nedolm.block_count", GGUF_TYPE_UINT32, arch["block_count"]),
        ("nedolm.feed_forward_length", GGUF_TYPE_UINT32, arch["feed_forward_length"]),
        ("nedolm.attention.head_count", GGUF_TYPE_UINT32, arch["head_count"]),
        ("nedolm.attention.head_count_kv", GGUF_TYPE_UINT32, arch["head_count_kv"]),
        ("nedolm.attention.layer_norm_rms_epsilon", GGUF_TYPE_FLOAT32, arch["rms_epsilon"]),
        ("nedolm.attention.sliding_window", GGUF_TYPE_UINT32, arch["sliding_window"]),
        ("nedolm.rope.dimension_count", GGUF_TYPE_UINT32, arch["head_dim"]),
        ("nedolm.rope.freq_base", GGUF_TYPE_FLOAT32, arch["rope_freq_base"]),
        ("nedolm.morph.layer_count", GGUF_TYPE_UINT32, arch["morph_layer_count"]),
        ("nedolm.morph.shared_width", GGUF_TYPE_UINT32, arch["morph_shared_width"]),
        ("nedolm.morph.root_width", GGUF_TYPE_UINT32, arch["morph_root_width"]),
        ("nedolm.morph.suffix_width", GGUF_TYPE_UINT32, arch["morph_suffix_width"]),
        ("nedolm.tie_word_embeddings", GGUF_TYPE_BOOL, arch["tie_word_embeddings"]),
        ("nedolm.source_checkpoint", GGUF_TYPE_STRING, checkpoint_name),
        ("nedolm.vocab_sha256", GGUF_TYPE_STRING, vocab_sha),
        ("tokenizer.ggml.model", GGUF_TYPE_STRING, "nedolm"),
        ("tokenizer.ggml.tokens", GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, tokens)),
        ("tokenizer.ggml.token_type", GGUF_TYPE_ARRAY, (GGUF_TYPE_INT32, token_types)),
        ("tokenizer.ggml.bos_token_id", GGUF_TYPE_UINT32, 1),
        ("tokenizer.ggml.eos_token_id", GGUF_TYPE_UINT32, 2),
        ("tokenizer.ggml.padding_token_id", GGUF_TYPE_UINT32, 0),
        ("tokenizer.ggml.add_bos_token", GGUF_TYPE_BOOL, False),
        ("tokenizer.ggml.add_eos_token", GGUF_TYPE_BOOL, False),
    ]
    if int(arch["vocab_size"]) >= 32002:
        metadata.extend([
            ("mercan.chat.message_start_token_id", GGUF_TYPE_UINT32, 32000),
            ("mercan.chat.message_end_token_id", GGUF_TYPE_UINT32, 32001),
            ("mercan.chat.stop_token_ids", GGUF_TYPE_ARRAY, (GGUF_TYPE_INT32, [32001, 2, 32000])),
            ("mercan.chat.role_system", GGUF_TYPE_STRING, "sistem"),
            ("mercan.chat.role_user", GGUF_TYPE_STRING, "kullanici"),
            ("mercan.chat.role_assistant", GGUF_TYPE_STRING, "asistan"),
        ])
    return metadata


def validate_arch(arch: Dict[str, object]) -> None:
    required = set(DEFAULT_ARCH)
    missing = sorted(required - set(arch))
    if missing:
        raise RuntimeError(f"architecture config missing keys: {missing}")
    if int(arch["morph_layer_count"]) > int(arch["block_count"]):
        raise RuntimeError("morph_layer_count cannot exceed block_count")
    widths = int(arch["morph_shared_width"]) + int(arch["morph_root_width"]) + int(arch["morph_suffix_width"])
    if widths != int(arch["feed_forward_length"]):
        raise RuntimeError(f"MorphFFN widths sum to {widths}, expected d_ff={arch['feed_forward_length']}")


def write_mercan(
    output: Path,
    tensors: List[Tuple[str, torch.Tensor, int]],
    metadata: List[Tuple[str, int, object]],
) -> Dict[str, object]:
    infos = []
    offset = 0
    for name, tensor, ggml_type in tensors:
        shape = tuple(int(x) for x in tensor.shape)
        itemsize = 4 if ggml_type == GGML_TYPE_F32 else 2
        offset = align_up(offset)
        nbytes = int(tensor.numel()) * itemsize
        infos.append((name, shape, offset, nbytes, tensor, ggml_type))
        offset += nbytes

    header = bytearray(b"GGUF") + struct.pack("<IQQ", GGUF_VERSION, len(infos), len(metadata))
    for item in metadata:
        header += pack_meta(*item)
    for name, shape, tensor_offset, _nbytes, _tensor, ggml_type in infos:
        header += pack_string(name)
        header += struct.pack("<I", len(shape))
        for dim in reversed(shape):
            header += struct.pack("<Q", dim)
        header += struct.pack("<I", ggml_type)
        header += struct.pack("<Q", tensor_offset)

    data_start = align_up(len(header))
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    with tmp.open("wb", buffering=16 << 20) as f:
        f.write(header)
        if f.tell() < data_start:
            f.write(b"\0" * (data_start - f.tell()))
        base = f.tell()
        for index, (name, _shape, tensor_offset, nbytes, tensor, ggml_type) in enumerate(infos, 1):
            target = base + tensor_offset
            if f.tell() < target:
                f.write(b"\0" * (target - f.tell()))
            dtype = torch.float32 if ggml_type == GGML_TYPE_F32 else torch.float16
            array = tensor.detach().cpu().to(dtype).contiguous().numpy()
            raw = array.tobytes(order="C")
            if len(raw) != nbytes:
                raise RuntimeError(f"tensor byte mismatch for {name}: {len(raw)} != {nbytes}")
            f.write(raw)
            if index % 20 == 0 or index == len(infos):
                print(f"WRITE {index}/{len(infos)} {name}", flush=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, output)

    with output.open("rb") as f:
        magic = f.read(4)
        version, tensor_count, kv_count = struct.unpack("<IQQ", f.read(20))
    if magic != b"GGUF" or version != GGUF_VERSION or tensor_count != len(infos):
        raise RuntimeError("post-write Mercan container validation failed")
    return {
        "format": "mercan",
        "format_version": 1,
        "physical_container": "GGUF-v3",
        "tensor_count": tensor_count,
        "metadata_count": kv_count,
        "size_bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_CHECKPOINT, help="PyTorch checkpoint file or HF model directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vocab-bin", type=Path, default=DEFAULT_VOCAB_BIN)
    parser.add_argument("--vocab-txt", type=Path, default=DEFAULT_VOCAB_TXT)
    parser.add_argument("--role-table", type=Path, default=DEFAULT_ROLE_TABLE)
    parser.add_argument("--arch-json", type=Path, help="optional JSON object overriding Mercan architecture fields")
    parser.add_argument("--model-name", default="Mercan-0.8B-SFT")
    args = parser.parse_args()

    arch = dict(DEFAULT_ARCH)
    if args.arch_json:
        arch.update(json.loads(args.arch_json.read_text("utf-8")))
    validate_arch(arch)

    vocab_bytes = args.vocab_bin.read_bytes()
    if vocab_bytes[:8] != b"NDSRF004":
        raise RuntimeError(f"wrong tokenizer asset magic: {vocab_bytes[:8]!r}")
    vocab_sha = sha256_bytes(vocab_bytes)
    if vocab_sha != EXPECTED_NDSRF004_SHA256:
        raise RuntimeError(f"unexpected NDSRF004 SHA256: {vocab_sha}")
    tokens, token_types = parse_vocab_txt(args.vocab_txt, int(arch["vocab_size"]))

    print(f"LOAD source={args.source}", flush=True)
    state, provenance = load_state_dict(args.source)
    tensors = map_tensors(state, arch, args.role_table)
    metadata = build_metadata(arch, vocab_bytes, vocab_sha, tokens, token_types, provenance, args.model_name)
    print(f"MAPPED tensors={len(tensors)} metadata={len(metadata)}", flush=True)

    result = write_mercan(args.output, tensors, metadata)
    result.update({
        "model_name": args.model_name,
        "source": str(args.source),
        "vocab_sha256": vocab_sha,
        "architecture": arch,
    })
    manifest = args.output.with_suffix(args.output.suffix + ".json")
    manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sums = args.output.parent / "SHA256SUMS"
    sums.write_text(f"{result['sha256']}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(f"MERCAN_EXPORT_OK {args.output}", flush=True)


if __name__ == "__main__":
    main()
