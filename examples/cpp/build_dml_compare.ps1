$ErrorActionPreference = "Stop"

# All intermediate files, temp files and caches stay inside the D-drive project
# folder. Nothing is written to C.
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # examples\cpp -> repo root
Set-Location $root
Write-Host "repo root: $root"

New-Item -ItemType Directory -Force -Path "$root\build\tmp" | Out-Null
New-Item -ItemType Directory -Force -Path "$root\build\native" | Out-Null
New-Item -ItemType Directory -Force -Path "$root\build\benchmarks" | Out-Null
New-Item -ItemType Directory -Force -Path "$root\examples\cpp\bin" | Out-Null

# Redirect MSVC temp dirs to the D-drive project folder.
$env:TMP = "$root\build\tmp"
$env:TEMP = "$root\build\tmp"

$ortRoot = "$root\build\third_party\onnxruntime-directml"
$ortInclude = "$ortRoot\build\native\include"
$ortNative = "$ortRoot\runtimes\win-x64\native"
Write-Host "ort include: $ortInclude -> $(Test-Path "$ortInclude\onnxruntime_cxx_api.h")"
Write-Host "ort native:  $ortNative -> $(Test-Path "$ortNative\onnxruntime.lib")"
if (!(Test-Path "$ortInclude\onnxruntime_cxx_api.h") -or !(Test-Path "$ortNative\onnxruntime.lib")) {
    throw "onnxruntime-directml not found under build\third_party\onnxruntime-directml. Download it first (nuget Microsoft.ML.OnnxRuntime.DirectML)."
}

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

$cmd = @(
    "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul",
    "cl /std:c++17 /EHsc /O2 /utf-8 native\aexrt_d3d12_runtime.cpp /LD /Febuild\native\aexrt_native_cpp.dll /Fobuild\native\aexrt_d3d12_runtime.obj /link d3d12.lib dxgi.lib d3dcompiler.lib",
    "cl /std:c++17 /EHsc /O2 /utf-8 examples\cpp\dml_compare.cpp /I native /I `"$ortInclude`" /Fe:examples\cpp\bin\dml_compare.exe /Fo:examples\cpp\bin\dml_compare.obj build\native\aexrt_native_cpp.lib /link d3d12.lib dxgi.lib d3dcompiler.lib `"$ortNative\onnxruntime.lib`"",
    "cl /std:c++17 /EHsc /O2 /utf-8 examples\cpp\conv_autotune.cpp /I native /Fe:examples\cpp\bin\conv_autotune.exe /Fo:examples\cpp\bin\conv_autotune.obj build\native\aexrt_native_cpp.lib /link d3d12.lib dxgi.lib d3dcompiler.lib",
    "cl /std:c++17 /EHsc /O2 /utf-8 examples\cpp\combo_repro.cpp /I native /Fe:examples\cpp\bin\combo_repro.exe /Fo:examples\cpp\bin\combo_repro.obj build\native\aexrt_native_cpp.lib /link d3d12.lib dxgi.lib d3dcompiler.lib"
) -join " && "

cmd /c $cmd
if ($LASTEXITCODE -ne 0) {
    throw "build failed with exit code $LASTEXITCODE"
}

Copy-Item build\native\aexrt_native_cpp.dll examples\cpp\bin\aexrt_native_cpp.dll -Force
foreach ($dll in @("onnxruntime.dll", "onnxruntime_providers_shared.dll")) {
    $source = Join-Path $ortNative $dll
    if (Test-Path $source) {
        Copy-Item $source examples\cpp\bin\ -Force
    }
}

Write-Host "built: examples\cpp\bin\dml_compare.exe"
Write-Host "usage: examples\cpp\bin\dml_compare.exe --list-adapters"
Write-Host "       examples\cpp\bin\dml_compare.exe --runs 100 --warmup 20"
