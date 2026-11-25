"""Gộp nhiều file thành 1 file .md để dán cho AI. Dùng: python scripts/make_bundle.py <out> <file1> <file2> ..."""
import sys
from pathlib import Path

out, files = sys.argv[1], sys.argv[2:]
buf = []
for f in files:
    p = Path(f)
    if not p.exists():
        print(f"BỎ QUA (không thấy): {f}")
        continue
    buf.append(f"\n\n===== FILE: {f} =====\n" + p.read_text(encoding="utf-8"))
Path(out).write_text("".join(buf).lstrip(), encoding="utf-8")
print(f"OK -> {out} ({len(buf)} file)")
