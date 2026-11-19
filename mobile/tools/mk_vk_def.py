"""Turn `dumpbin /exports vulkan-1.dll` output into a .def file for lib.exe."""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
names = []
for line in open(src, encoding="utf-8", errors="replace"):
    m = re.match(r"\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)", line)
    if m and m.group(1).startswith("vk"):
        names.append(m.group(1))
if not names:
    raise SystemExit("no vk* exports found in " + src)
with open(dst, "w", encoding="ascii", newline="\n") as f:
    f.write("LIBRARY vulkan-1.dll\nEXPORTS\n")
    for n in names:
        f.write("    %s\n" % n)
print(f"{dst}: {len(names)} exports")
