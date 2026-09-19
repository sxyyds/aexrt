$ErrorActionPreference = "Stop"

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (!(Test-Path $vswhere)) {
    throw "vswhere.exe not found. Install Visual Studio Build Tools or run from a Developer PowerShell."
}

$vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vsInstall) {
    throw "MSVC C++ tools not found."
}

$vsDevCmd = Join-Path $vsInstall "Common7\Tools\VsDevCmd.bat"
if (!(Test-Path $vsDevCmd)) {
    throw "VsDevCmd.bat not found under $vsInstall"
}

New-Item -ItemType Directory -Force -Path build\native | Out-Null
New-Item -ItemType Directory -Force -Path examples\cpp\bin | Out-Null

$cmd = @(
    "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul",
    "cl /std:c++17 /EHsc /O2 native\aexrt_d3d12_runtime.cpp /LD /Febuild\native\aexrt_native_cpp.dll /Fobuild\native\aexrt_d3d12_runtime.obj /link d3d12.lib dxgi.lib d3dcompiler.lib",
    "cl /std:c++17 /EHsc /O2 examples\cpp\native_relu_pure.cpp /I native /Fe:examples\cpp\bin\native_relu_pure.exe /Fo:examples\cpp\bin\native_relu_pure.obj build\native\aexrt_native_cpp.lib",
    "cl /std:c++17 /EHsc /O2 examples\cpp\native_relu_graph.cpp /I native /Fe:examples\cpp\bin\native_relu_graph.exe /Fo:examples\cpp\bin\native_relu_graph.obj build\native\aexrt_native_cpp.lib",
    "cl /std:c++17 /EHsc /O2 examples\cpp\native_relu_gelu_graph.cpp /I native /Fe:examples\cpp\bin\native_relu_gelu_graph.exe /Fo:examples\cpp\bin\native_relu_gelu_graph.obj build\native\aexrt_native_cpp.lib",
    "cl /std:c++17 /EHsc /O2 examples\cpp\native_add_relu_graph.cpp /I native /Fe:examples\cpp\bin\native_add_relu_graph.exe /Fo:examples\cpp\bin\native_add_relu_graph.obj build\native\aexrt_native_cpp.lib",
    "cl /std:c++17 /EHsc /O2 examples\cpp\native_dynamic_elementwise_graph.cpp /I native /Fe:examples\cpp\bin\native_dynamic_elementwise_graph.exe /Fo:examples\cpp\bin\native_dynamic_elementwise_graph.obj build\native\aexrt_native_cpp.lib",
    "cl /std:c++17 /EHsc /O2 examples\cpp\native_three_input_graph.cpp /I native /Fe:examples\cpp\bin\native_three_input_graph.exe /Fo:examples\cpp\bin\native_three_input_graph.obj build\native\aexrt_native_cpp.lib",
    "cl /std:c++17 /EHsc /O2 examples\cpp\native_conv_silu.cpp /I native /Fe:examples\cpp\bin\native_conv_silu.exe /Fo:examples\cpp\bin\native_conv_silu.obj build\native\aexrt_native_cpp.lib",
    "cl /std:c++17 /EHsc /O2 examples\cpp\native_load_graph_json.cpp /I native /Fe:examples\cpp\bin\native_load_graph_json.exe /Fo:examples\cpp\bin\native_load_graph_json.obj build\native\aexrt_native_cpp.lib",
    "cl /std:c++17 /EHsc /O2 examples\cpp\native_yolo_package.cpp /I native /Fe:examples\cpp\bin\native_yolo_package.exe /Fo:examples\cpp\bin\native_yolo_package.obj build\native\aexrt_native_cpp.lib",
    "cl /std:c++17 /EHsc /O2 examples\cpp\native_dxgi_yolo_overlay.cpp /I native /Fe:examples\cpp\bin\native_dxgi_yolo_overlay.exe /Fo:examples\cpp\bin\native_dxgi_yolo_overlay.obj build\native\aexrt_native_cpp.lib /link d3d11.lib dxgi.lib user32.lib gdi32.lib",
    "cl /std:c++17 /EHsc /O2 /utf-8 examples\cpp\native_imgui_yolo_browser.cpp third_party\imgui\imgui.cpp third_party\imgui\imgui_draw.cpp third_party\imgui\imgui_tables.cpp third_party\imgui\imgui_widgets.cpp third_party\imgui\backends\imgui_impl_win32.cpp third_party\imgui\backends\imgui_impl_dx11.cpp /I native /I third_party\imgui /I third_party\imgui\backends /Fe:examples\cpp\bin\native_imgui_yolo_browser.exe /Fo:examples\cpp\bin\ build\native\aexrt_native_cpp.lib /link d3d11.lib dxgi.lib user32.lib gdi32.lib comdlg32.lib shell32.lib"
) -join " && "

cmd /c $cmd
Copy-Item build\native\aexrt_native_cpp.dll examples\cpp\bin\aexrt_native_cpp.dll -Force

$dxcRoot = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"
$dxcDir = Get-ChildItem $dxcRoot -Directory -ErrorAction SilentlyContinue |
    Where-Object {
        (Test-Path (Join-Path $_.FullName "x64\dxcompiler.dll")) -and
        (Test-Path (Join-Path $_.FullName "x64\dxil.dll"))
    } |
    Sort-Object { [version]$_.Name } -Descending |
    Select-Object -First 1
if ($dxcDir) {
    $dxcBin = Join-Path $dxcDir.FullName "x64"
    foreach ($target in @("build\native", "examples\cpp\bin")) {
        Copy-Item (Join-Path $dxcBin "dxcompiler.dll") $target -Force
        Copy-Item (Join-Path $dxcBin "dxil.dll") $target -Force
    }
} else {
    Write-Warning "DXC runtime DLLs were not found; SM5 FP32 remains available but SM6 native FP16 is disabled."
}
