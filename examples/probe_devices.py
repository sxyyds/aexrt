import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aexrt import HostDevice, NativeD3D12Device


host = HostDevice().info()
d3d12 = NativeD3D12Device.probe()

print("host:", host)
print("native_d3d12:", d3d12)
