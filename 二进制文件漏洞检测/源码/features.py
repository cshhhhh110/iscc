"""Feature extraction utilities for binary files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
import pefile

from disasm_features import extract_disasm_features, get_disasm_feature_names


PathLike = Union[str, Path]

ASCII_RE = re.compile(rb"[ -~]{4,}")

SECTION_NAMES = [
    ".text",
    ".data",
    ".rdata",
    ".bss",
    ".idata",
    ".edata",
    ".rsrc",
    ".reloc",
    ".tls",
    ".pdata",
    ".xdata",
    ".CRT",
    ".didat",
    ".init",
    ".fini",
    ".code",
    ".rodata",
    ".upx0",
    ".upx1",
    ".gfids",
    ".sdata",
    ".sbss",
]

DLL_NAMES = [
    "kernel32.dll",
    "user32.dll",
    "advapi32.dll",
    "ws2_32.dll",
    "msvcrt.dll",
    "ntdll.dll",
    "shell32.dll",
    "ole32.dll",
    "oleaut32.dll",
    "gdi32.dll",
    "crypt32.dll",
    "wininet.dll",
    "urlmon.dll",
    "iphlpapi.dll",
    "shlwapi.dll",
    "comdlg32.dll",
    "setupapi.dll",
    "version.dll",
    "winmm.dll",
    "mpr.dll",
]

API_NAMES = [
    "createfilea",
    "createfilew",
    "readfile",
    "writefile",
    "closehandle",
    "deletefilea",
    "deletefilew",
    "movefilea",
    "movefilew",
    "copyfilea",
    "copyfilew",
    "getprocaddress",
    "loadlibrarya",
    "loadlibraryw",
    "virtualalloc",
    "virtualprotect",
    "virtualfree",
    "heapalloc",
    "heapfree",
    "malloc",
    "calloc",
    "realloc",
    "free",
    "memcpy",
    "memmove",
    "memset",
    "strcpy",
    "strncpy",
    "strcat",
    "strncat",
    "sprintf",
    "snprintf",
    "vsprintf",
    "scanf",
    "sscanf",
    "gets",
    "fopen",
    "fread",
    "fwrite",
    "fclose",
    "system",
    "winexec",
    "createprocessa",
    "createprocessw",
    "shell_executea",
    "shell_executew",
    "internetopena",
    "internetopenw",
    "internetopenurla",
    "internetopenurlw",
    "socket",
    "connect",
    "send",
    "recv",
    "accept",
    "bind",
    "listen",
    "regopenkeya",
    "regopenkeyw",
    "regsetvaluea",
    "regsetvaluew",
    "regcreatekeya",
    "regcreatekeyw",
]

STRING_KEYWORDS = [
    "http://",
    "https://",
    "ftp://",
    "cmd.exe",
    "powershell",
    "system(",
    "createprocess",
    "virtualalloc",
    "virtualprotect",
    "winexec",
    "shell_execute",
    "download",
    "upload",
    "\\windows\\",
    "c:\\",
    "/bin/",
    "%s",
    "%d",
    "%x",
    "%n",
    "../",
    "..\\",
    ".exe",
    ".dll",
    "eval(",
    "exec(",
]

BASE_FEATURES = [
    "file_size",
    "byte_entropy",
    "byte_mean",
    "byte_std",
    "byte_zero_ratio",
    "byte_printable_ratio",
    "byte_highbit_ratio",
    "byte_unique_ratio",
    "pe_valid",
    "pe_machine",
    "pe_magic",
    "pe_number_of_sections",
    "pe_timestamp",
    "pe_characteristics",
    "pe_subsystem",
    "pe_dll_characteristics",
    "pe_entry_point",
    "pe_image_base",
    "pe_section_alignment",
    "pe_file_alignment",
    "pe_size_of_image",
    "pe_size_of_headers",
    "pe_checksum",
    "pe_number_of_rva_and_sizes",
    "pe_size_of_code",
    "pe_size_of_initialized_data",
    "pe_size_of_uninitialized_data",
    "pe_base_of_code",
    "pe_base_of_data",
    "pe_num_import_dlls",
    "pe_num_imports",
    "pe_num_exports",
    "section_count",
    "section_raw_size_sum",
    "section_raw_size_mean",
    "section_raw_size_std",
    "section_virtual_size_sum",
    "section_virtual_size_mean",
    "section_virtual_size_std",
    "section_entropy_mean",
    "section_entropy_std",
    "section_entropy_max",
    "section_entropy_min",
    "section_exec_count",
    "section_write_count",
    "section_read_count",
    "section_discardable_count",
    "section_empty_count",
    "section_name_matches",
    "string_count",
    "string_total_length",
    "string_mean_length",
    "string_max_length",
    "string_unique_ratio",
    "string_url_count",
    "string_ip_count",
    "string_path_count",
    "string_format_count",
    "string_cmd_count",
]


def get_feature_columns() -> List[str]:
    columns: List[str] = list(BASE_FEATURES)
    columns.extend(f"byte_{i:03d}" for i in range(256))
    columns.extend(f"section_name_{name.lower().replace('.', '').replace('/', '_')}" for name in SECTION_NAMES)
    columns.extend(f"dll_{name.replace('.', '_')}" for name in DLL_NAMES)
    columns.extend(f"api_{name}" for name in API_NAMES)
    columns.extend(f"keyword_{name}" for name in STRING_KEYWORDS)
    columns.extend(get_disasm_feature_names())
    return columns


FEATURE_COLUMNS = get_feature_columns()


def _safe_decode(value: bytes) -> str:
    return value.decode("utf-8", errors="ignore").strip("\x00").strip().lower()


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def _section_entropy(section: pefile.SectionStructure) -> float:
    try:
        return _entropy(section.get_data())
    except Exception:
        return 0.0


def _init_feature_dict() -> Dict[str, float]:
    return {name: 0.0 for name in FEATURE_COLUMNS}


def _count_patterns(text: str, patterns: Iterable[str]) -> Dict[str, float]:
    lowered = text.lower()
    return {f"keyword_{pattern}": float(lowered.count(pattern)) for pattern in patterns}


def extract_features(binary_path: PathLike) -> Dict[str, float]:
    """Extract deterministic features from a single binary file."""

    path = Path(binary_path)
    raw = path.read_bytes()
    feats = _init_feature_dict()

    feats["file_size"] = float(len(raw))
    feats["byte_entropy"] = _entropy(raw)
    if raw:
        arr = np.frombuffer(raw, dtype=np.uint8)
        feats["byte_mean"] = float(arr.mean())
        feats["byte_std"] = float(arr.std())
        feats["byte_zero_ratio"] = float(np.mean(arr == 0))
        feats["byte_printable_ratio"] = float(np.mean((arr >= 32) & (arr <= 126)))
        feats["byte_highbit_ratio"] = float(np.mean(arr >= 128))
        feats["byte_unique_ratio"] = float(len(np.unique(arr)) / 256.0)
        hist = np.bincount(arr, minlength=256).astype(np.float32)
        hist /= float(len(arr))
        for i, value in enumerate(hist):
            feats[f"byte_{i:03d}"] = float(value)

    strings = [m.decode("ascii", errors="ignore") for m in ASCII_RE.findall(raw)]
    if strings:
        total_length = sum(len(s) for s in strings)
        feats["string_count"] = float(len(strings))
        feats["string_total_length"] = float(total_length)
        feats["string_mean_length"] = float(total_length / len(strings))
        feats["string_max_length"] = float(max(len(s) for s in strings))
        feats["string_unique_ratio"] = float(len(set(strings)) / len(strings))
        joined = "\n".join(strings)
        lowered = joined.lower()
        feats["string_url_count"] = float(lowered.count("http://") + lowered.count("https://") + lowered.count("ftp://"))
        feats["string_ip_count"] = float(len(re.findall(r"(?:\d{1,3}\.){3}\d{1,3}", joined)))
        feats["string_path_count"] = float(
            lowered.count("\\windows\\")
            + lowered.count("c:\\")
            + lowered.count("/bin/")
            + lowered.count("../")
            + lowered.count("..\\")
        )
        feats["string_format_count"] = float(lowered.count("%s") + lowered.count("%d") + lowered.count("%x") + lowered.count("%n"))
        feats["string_cmd_count"] = float(
            lowered.count("cmd.exe")
            + lowered.count("powershell")
            + lowered.count("system(")
            + lowered.count("createprocess")
            + lowered.count("virtualalloc")
            + lowered.count("virtualprotect")
            + lowered.count("winexec")
            + lowered.count("shell_execute")
            + lowered.count("eval(")
            + lowered.count("exec(")
        )
        for key, value in _count_patterns(joined, STRING_KEYWORDS).items():
            feats[key] = value

    # Disassembly features (v2.3)
    disasm_feats = extract_disasm_features(path)
    feats.update(disasm_feats)

    try:
        pe = pefile.PE(data=raw, fast_load=True)
        feats["pe_valid"] = 1.0
        try:
            pe.parse_data_directories()
        except Exception:
            pass

        file_header = pe.FILE_HEADER
        optional = getattr(pe, "OPTIONAL_HEADER", None)
        feats["pe_machine"] = float(getattr(file_header, "Machine", 0))
        feats["pe_number_of_sections"] = float(getattr(file_header, "NumberOfSections", 0))
        feats["pe_timestamp"] = float(getattr(file_header, "TimeDateStamp", 0))
        feats["pe_characteristics"] = float(getattr(file_header, "Characteristics", 0))
        feats["pe_magic"] = float(getattr(optional, "Magic", 0) if optional else 0)
        feats["pe_subsystem"] = float(getattr(optional, "Subsystem", 0) if optional else 0)
        feats["pe_dll_characteristics"] = float(getattr(optional, "DllCharacteristics", 0) if optional else 0)
        feats["pe_entry_point"] = float(getattr(optional, "AddressOfEntryPoint", 0) if optional else 0)
        feats["pe_image_base"] = float(getattr(optional, "ImageBase", 0) if optional else 0)
        feats["pe_section_alignment"] = float(getattr(optional, "SectionAlignment", 0) if optional else 0)
        feats["pe_file_alignment"] = float(getattr(optional, "FileAlignment", 0) if optional else 0)
        feats["pe_size_of_image"] = float(getattr(optional, "SizeOfImage", 0) if optional else 0)
        feats["pe_size_of_headers"] = float(getattr(optional, "SizeOfHeaders", 0) if optional else 0)
        feats["pe_checksum"] = float(getattr(optional, "CheckSum", 0) if optional else 0)
        feats["pe_number_of_rva_and_sizes"] = float(getattr(optional, "NumberOfRvaAndSizes", 0) if optional else 0)
        feats["pe_size_of_code"] = float(getattr(optional, "SizeOfCode", 0) if optional else 0)
        feats["pe_size_of_initialized_data"] = float(getattr(optional, "SizeOfInitializedData", 0) if optional else 0)
        feats["pe_size_of_uninitialized_data"] = float(getattr(optional, "SizeOfUninitializedData", 0) if optional else 0)
        feats["pe_base_of_code"] = float(getattr(optional, "BaseOfCode", 0) if optional else 0)
        feats["pe_base_of_data"] = float(getattr(optional, "BaseOfData", 0) if optional else 0)

        raw_sizes: List[float] = []
        virt_sizes: List[float] = []
        entropies: List[float] = []
        exec_count = 0
        write_count = 0
        read_count = 0
        discardable_count = 0
        empty_count = 0
        section_name_matches = 0

        for section in pe.sections:
            raw_size = float(getattr(section, "SizeOfRawData", 0))
            virt_size = float(getattr(section, "Misc_VirtualSize", 0))
            raw_sizes.append(raw_size)
            virt_sizes.append(virt_size)
            entropies.append(_section_entropy(section))

            characteristics = int(getattr(section, "Characteristics", 0))
            if characteristics & 0x20000000:
                exec_count += 1
            if characteristics & 0x80000000:
                write_count += 1
            if characteristics & 0x40000000:
                read_count += 1
            if characteristics & 0x02000000:
                discardable_count += 1
            if raw_size == 0:
                empty_count += 1

            section_name = _safe_decode(getattr(section, "Name", b""))
            normalized = section_name.lower()
            for name in SECTION_NAMES:
                normalized_key = name.lower().replace(".", "").replace("/", "_")
                if normalized == name.lower():
                    feats[f"section_name_{normalized_key}"] += 1.0
                    section_name_matches += 1

        feats["section_count"] = float(len(raw_sizes))
        feats["section_raw_size_sum"] = float(sum(raw_sizes))
        feats["section_raw_size_mean"] = float(np.mean(raw_sizes)) if raw_sizes else 0.0
        feats["section_raw_size_std"] = float(np.std(raw_sizes)) if raw_sizes else 0.0
        feats["section_virtual_size_sum"] = float(sum(virt_sizes))
        feats["section_virtual_size_mean"] = float(np.mean(virt_sizes)) if virt_sizes else 0.0
        feats["section_virtual_size_std"] = float(np.std(virt_sizes)) if virt_sizes else 0.0
        feats["section_entropy_mean"] = float(np.mean(entropies)) if entropies else 0.0
        feats["section_entropy_std"] = float(np.std(entropies)) if entropies else 0.0
        feats["section_entropy_max"] = float(np.max(entropies)) if entropies else 0.0
        feats["section_entropy_min"] = float(np.min(entropies)) if entropies else 0.0
        feats["section_exec_count"] = float(exec_count)
        feats["section_write_count"] = float(write_count)
        feats["section_read_count"] = float(read_count)
        feats["section_discardable_count"] = float(discardable_count)
        feats["section_empty_count"] = float(empty_count)
        feats["section_name_matches"] = float(section_name_matches)

        import_dlls = []
        import_apis = []
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = _safe_decode(getattr(entry, "dll", b""))
                if dll_name:
                    import_dlls.append(dll_name)
                    normalized_dll = dll_name.lower()
                    for name in DLL_NAMES:
                        if normalized_dll == name:
                            feats[f"dll_{name.replace('.', '_')}"] += 1.0
                for imp in getattr(entry, "imports", []):
                    imp_name = _safe_decode(getattr(imp, "name", b""))
                    if imp_name:
                        import_apis.append(imp_name)
                        normalized_api = imp_name.lower().split("@", 1)[0]
                        for name in API_NAMES:
                            if normalized_api == name:
                                feats[f"api_{name}"] += 1.0
        feats["pe_num_import_dlls"] = float(len(set(import_dlls)))
        feats["pe_num_imports"] = float(len(import_apis))
        feats["pe_num_exports"] = float(len(getattr(getattr(pe, "DIRECTORY_ENTRY_EXPORT", None), "symbols", []) or []))
    except Exception:
        feats["pe_valid"] = 0.0

    return feats
