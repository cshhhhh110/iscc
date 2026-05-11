"""Capstone disassembly feature extraction for binary vulnerability detection (v2.4)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pefile
from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64

# x86 registers to track
X86_REGS = [
    "eax", "ebx", "ecx", "edx", "esi", "edi", "esp", "ebp",
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rsp", "rbp",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
    "eip", "rip",
]
REG_RE = re.compile(r'\b(' + '|'.join(X86_REGS) + r')\b', re.IGNORECASE)
# Function prologue patterns (consecutive opcode pairs)
PROLOGUE_PATTERNS = [
    ("push", "mov"),   # push rbp; mov rbp, rsp
    ("push", "sub"),   # push rbp; sub rsp, X
    ("mov", "push"),   # mov [rsp+X], ...; push ...
]

# Common x86 opcodes (top ~150 by frequency in PE binaries)
OPCODE_VOCAB = [
    "mov", "push", "pop", "call", "lea", "add", "sub", "cmp", "jmp", "je",
    "jne", "xor", "test", "and", "or", "ret", "nop", "inc", "dec", "jz",
    "jnz", "jg", "jl", "jge", "jle", "ja", "jb", "jae", "jbe", "jns",
    "js", "jp", "jnp", "jo", "jno", "shl", "shr", "sar", "rol", "ror",
    "imul", "idiv", "div", "mul", "neg", "not", "adc", "sbb", "xchg",
    "movzx", "movsx", "cdq", "cqo", "cdqe", "cwd", "cbw", "cwde",
    "leave", "int3", "int", "syscall", "fld", "fstp", "fst", "fadd",
    "fsub", "fmul", "fdiv", "fxch", "fcomp", "fucom", "fldz", "fld1",
    "rep", "repne", "repe", "loop", "loope", "loopne",
    "cmov", "cmove", "cmovne", "cmovg", "cmovl", "cmovge", "cmovle",
    "cmova", "cmovb", "cmovae", "cmovbe", "cmovs", "cmovns",
    "sete", "setne", "setg", "setl", "setge", "setle", "seta", "setb",
    "bt", "bts", "btr", "btc", "bsf", "bsr",
    "std", "cld", "stos", "lods", "movs", "cmps", "scas",
    "pushf", "popf", "pusha", "popa", "lahf", "sahf",
    "rdtsc", "cpuid", "pause", "lfence", "sfence", "mfence",
    "lock", "xadd", "cmpxchg",
    "pshufd", "movdqa", "movdqu", "padd", "psub", "pmul", "pxor", "pand",
    "paddd", "psubd", "pmulld", "pxor", "pcmpeq", "pcmpgt",
]

# Explicitly define bigram pairs to track (most discriminative for vuln detection)
OPCODE_BIGRAMS = [
    # Stack manipulation patterns
    "push|push", "push|call", "push|mov", "pop|pop", "pop|ret",
    # Function prologue/epilogue
    "push|mov", "mov|sub", "sub|mov", "mov|lea", "leave|ret",
    # Memory access patterns
    "mov|mov", "mov|lea", "mov|cmp", "mov|test", "mov|xor",
    "lea|mov", "lea|call",
    # Control flow
    "cmp|je", "cmp|jne", "cmp|jg", "cmp|jl", "cmp|jmp",
    "test|je", "test|jne", "test|jz",
    "call|mov", "call|test", "call|cmp", "call|push",
    # Arithmetic followed by control flow
    "add|cmp", "sub|cmp", "add|jmp", "inc|cmp",
    # Buffer manipulation (relevant for overflow detection)
    "mov|add", "lea|add", "mov|inc", "lea|inc",
    # Loop patterns
    "inc|cmp", "cmp|jl", "dec|jne",
    # XOR zeroing patterns
    "xor|cmp", "xor|test",
    # Repeated instructions (string ops, memcpy patterns)
    "rep|movs", "rep|stos", "mov|rep",
]


def _get_text_section_offset(pe: pefile.PE) -> Optional[Tuple[int, int, bool]]:
    """Return (offset, size, is_64bit) for the .text section, or None."""
    is_64 = pe.FILE_HEADER.Machine == 0x8664 or (hasattr(pe, "OPTIONAL_HEADER") and
            getattr(pe.OPTIONAL_HEADER, "Magic", 0) == 0x20b)
    for section in pe.sections:
        name = getattr(section, "Name", b"").decode("utf-8", errors="ignore").strip("\x00").strip()
        if name == ".text":
            return (section.PointerToRawData, section.SizeOfRawData, is_64)
    return None


def _disassemble(raw_data: bytes, offset: int, size: int, is_64: bool):
    """Disassemble a code section and return (opcodes, op_strs, sizes, bytes_list)."""
    mode = CS_MODE_64 if is_64 else CS_MODE_32
    md = Cs(CS_ARCH_X86, mode)
    md.detail = False
    code = raw_data[offset:offset + min(size, len(raw_data) - offset)]
    opcodes: List[str] = []
    op_strs: List[str] = []
    sizes: List[int] = []
    instr_bytes: List[bytes] = []
    max_instrs = 50000
    try:
        for i, (addr, sz, mnemonic, op_str) in enumerate(md.disasm_lite(code, 0)):
            if i >= max_instrs:
                break
            opcodes.append(mnemonic.strip().lower())
            op_strs.append(op_str.strip().lower())
            sizes.append(sz)
            instr_bytes.append(code[addr:addr + sz])
    except Exception:
        pass
    return opcodes, op_strs, sizes, instr_bytes


def _opcode_stats(opcodes: List[str]) -> Dict[str, float]:
    """Compute instruction-level statistics."""
    feats: Dict[str, float] = {}
    n = max(len(opcodes), 1)
    feats["disasm_total_instrs"] = float(len(opcodes))
    feats["disasm_unique_opcodes"] = float(len(set(opcodes)))
    feats["disasm_opcode_entropy"] = 0.0
    if len(opcodes) > 1:
        counts = np.bincount([hash(o) % 1000 for o in opcodes], minlength=1000).astype(np.float64)
        probs = counts[counts > 0] / counts.sum()
        feats["disasm_opcode_entropy"] = float(-(probs * np.log2(probs + 1e-12)).sum())
    # Branch density
    branch_ops = {"jmp", "je", "jne", "jg", "jl", "jge", "jle", "ja", "jb", "jae", "jbe",
                  "jz", "jnz", "jns", "js", "jp", "jnp", "jo", "jno", "call", "ret", "loop"}
    branches = sum(1 for o in opcodes if o in branch_ops)
    feats["disasm_branch_ratio"] = branches / n
    # Memory-intensive operations (potential vulnerability indicators)
    mem_ops = {"mov", "lea", "push", "pop", "xchg", "stos", "lods", "movs", "cmps", "scas"}
    mem_count = sum(1 for o in opcodes if o in mem_ops)
    feats["disasm_mem_ratio"] = mem_count / n
    return feats


def _opcode_unigrams(opcodes: List[str]) -> Dict[str, float]:
    """Count opcode unigrams."""
    feats: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for o in opcodes:
        counts[o] = counts.get(o, 0) + 1
    total = max(len(opcodes), 1)
    for op in OPCODE_VOCAB:
        feats[f"disasm_op_{op}"] = counts.get(op, 0) / total
    return feats


def _opcode_bigrams(opcodes: List[str]) -> Dict[str, float]:
    """Count predefined bigram pairs."""
    feats: Dict[str, float] = {}
    pair_counts: Dict[str, int] = {}
    for i in range(len(opcodes) - 1):
        pair = f"{opcodes[i]}|{opcodes[i+1]}"
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    total = max(len(opcodes) - 1, 1)
    for bg in OPCODE_BIGRAMS:
        feats[f"disasm_bg_{bg}"] = pair_counts.get(bg, 0) / total
    return feats


# ---- v2.4 new features ----


def _register_usage(op_strs: List[str]) -> Dict[str, float]:
    """Register frequency heatmap."""
    feats: Dict[str, float] = {}
    all_ops = " ".join(op_strs)
    total = max(len(op_strs), 1)
    for reg in X86_REGS:
        count = len(REG_RE.findall(all_ops))
        key_safe = reg.replace(" ", "_")
        feats[f"disasm_reg_{key_safe}"] = count / total
    return feats


def _function_stats(opcodes: List[str]) -> Dict[str, float]:
    """Function boundary detection via prologue heuristics."""
    feats: Dict[str, float] = {}
    n = len(opcodes)
    feats["disasm_func_count"] = 0.0
    feats["disasm_func_avg_instrs"] = 0.0
    feats["disasm_func_max_instrs"] = 0.0

    # Detect prologues: push+matching opcode pairs at likely function starts
    func_starts: List[int] = []
    i = 0
    while i < n - 1:
        pair = (opcodes[i], opcodes[i + 1])
        if pair in PROLOGUE_PATTERNS:
            # A function start likely has push rbp pattern and we're at a clean boundary
            # Only count if we're not in the middle of a known function
            if not func_starts or i - func_starts[-1] > 10:  # min 10 instrs between functions
                func_starts.append(i)
        i += 1

    if func_starts:
        feats["disasm_func_count"] = float(len(func_starts))
        func_sizes = []
        for j in range(len(func_starts)):
            start = func_starts[j]
            end = func_starts[j + 1] if j + 1 < len(func_starts) else n
            func_sizes.append(end - start)
        feats["disasm_func_avg_instrs"] = float(np.mean(func_sizes)) if func_sizes else 0.0
        feats["disasm_func_max_instrs"] = float(max(func_sizes)) if func_sizes else 0.0

    return feats


def _operand_stats(opcodes: List[str], op_strs: List[str]) -> Dict[str, float]:
    """Operand pattern statistics."""
    feats: Dict[str, float] = {}
    n = max(len(opcodes), 1)

    # Stack operations: push, pop, or instructions referencing esp/rsp/ebp/rbp
    stack_ops = {"push", "pop", "pusha", "popa", "pushf", "popf", "enter", "leave"}
    stack_count = sum(1 for o in opcodes if o in stack_ops)
    # Add reg references to stack pointers
    rsp_ebp = 0
    for ops in op_strs:
        if "esp" in ops or "rsp" in ops or "ebp" in ops or "rbp" in ops:
            rsp_ebp += 1
    feats["disasm_stack_op_ratio"] = (stack_count + rsp_ebp * 0.3) / n

    # Immediate operand density
    imm_count = 0
    for ops in op_strs:
        # Has immediate if contains hex or decimal numbers
        if re.search(r'0x[0-9a-f]+|\b[0-9]+\b', ops):
            imm_count += 1
    feats["disasm_imm_ratio"] = imm_count / n

    # Memory operand density (instructions accessing [mem])
    mem_ref_count = sum(1 for ops in op_strs if "[" in ops or "ptr" in ops)
    feats["disasm_memref_ratio"] = mem_ref_count / n

    return feats


def _instr_length_stats(sizes: List[int], instr_bytes: List[bytes]) -> Dict[str, float]:
    """Instruction byte-length statistics."""
    feats: Dict[str, float] = {}
    if not sizes:
        return feats
    feats["disasm_avg_instr_bytes"] = float(np.mean(sizes))
    feats["disasm_std_instr_bytes"] = float(np.std(sizes))
    # Long instruction ratio (>4 bytes = complex addressing or prefix-heavy)
    long_count = sum(1 for s in sizes if s > 4)
    feats["disasm_long_instr_ratio"] = long_count / len(sizes)
    # Prefix-heavy ratio (>6 bytes)
    very_long = sum(1 for s in sizes if s > 6)
    feats["disasm_verylong_instr_ratio"] = very_long / len(sizes)
    return feats


def extract_disasm_features(binary_path: Path) -> Dict[str, float]:
    """Extract Capstone disassembly features from a PE binary."""
    feats: Dict[str, float] = {}
    path = Path(binary_path)
    try:
        raw = path.read_bytes()
    except OSError:
        return feats

    try:
        pe = pefile.PE(data=raw, fast_load=True)
        section_info = _get_text_section_offset(pe)
        pe.close()
    except Exception:
        return feats

    if section_info is None:
        return feats

    offset, size, is_64 = section_info
    if size == 0 or offset <= 0:
        return feats

    opcodes, op_strs, sizes, instr_bytes = _disassemble(raw, offset, size, is_64)
    if not opcodes:
        return feats

    feats.update(_opcode_stats(opcodes))
    feats.update(_opcode_unigrams(opcodes))
    feats.update(_opcode_bigrams(opcodes))
    feats.update(_register_usage(op_strs))
    feats.update(_function_stats(opcodes))
    feats.update(_operand_stats(opcodes, op_strs))
    feats.update(_instr_length_stats(sizes, instr_bytes))
    return feats


def get_disasm_feature_names() -> List[str]:
    """Return the ordered list of disassembly feature column names."""
    names: List[str] = []
    # Stats
    names.append("disasm_total_instrs")
    names.append("disasm_unique_opcodes")
    names.append("disasm_opcode_entropy")
    names.append("disasm_branch_ratio")
    names.append("disasm_mem_ratio")
    # Unigrams
    for op in OPCODE_VOCAB:
        names.append(f"disasm_op_{op}")
    # Bigrams
    for bg in OPCODE_BIGRAMS:
        names.append(f"disasm_bg_{bg}")
    # Registers
    for reg in X86_REGS:
        key_safe = reg.replace(" ", "_")
        names.append(f"disasm_reg_{key_safe}")
    # Functions
    names.append("disasm_func_count")
    names.append("disasm_func_avg_instrs")
    names.append("disasm_func_max_instrs")
    # Operands
    names.append("disasm_stack_op_ratio")
    names.append("disasm_imm_ratio")
    names.append("disasm_memref_ratio")
    # Instr lengths
    names.append("disasm_avg_instr_bytes")
    names.append("disasm_std_instr_bytes")
    names.append("disasm_long_instr_ratio")
    names.append("disasm_verylong_instr_ratio")
    return names
