"""Kiểm tra chất lượng Step 4 (role_lines.json): kiểm tra sót câu, lọt chữ Hán, phân bố vai và in mốc chính."""
import json
import re
import sys
from pathlib import Path

p = Path(sys.argv[1])
data = json.loads(p.read_text(encoding="utf-8"))
roles = {}
empties = []
cjk_leaks = []
cjk_re = re.compile(r"[\u4e00-\u9fff]")

for x in data:
    roles[x["voice_role"]] = roles.get(x["voice_role"], 0) + 1
    for k in ("vi", "en", "ja", "ko"):
        v = x.get(k, "").strip()
        if not v:
            empties.append((x["id"], k))
        if k in ("vi", "en", "ko") and cjk_re.search(v):
            cjk_leaks.append((x["id"], k, v))

print("TOTAL_LINES:", len(data))
print("ROLES:", roles)
print("EMPTIES:", len(empties), empties[:5])
print("CJK_LEAKS (vi/en/ko):", len(cjk_leaks), cjk_leaks[:10])
print("SPAN:", data[0]["start"], "->", data[-1]["end"])
for frac in (0.0, 0.18, 0.36, 0.55, 0.75, 0.92, 0.99):
    idx = min(len(data) - 1, int(frac * len(data)))
    r = data[idx]
    print(f"[{r['id']}] {r['start']} ({r['voice_role']}) ZH={r['zh']} | VI={r['vi']}")
