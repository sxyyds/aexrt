$ErrorActionPreference = "Stop"

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (!(Test-Path $vswhere)) {
    throw "vswhere.exe not found. Install Visual Studio Build Tools or run cl from a Developer PowerShell."
}

$vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vsInstall) {
    throw "MSVC C++ tools not found."
}

$vsDevCmd = Join-Path $vsInstall "Common7\Tools\VsDevCmd.bat"
if (!(Test-Path $vsDevCmd)) {
    throw "VsDevCmd.bat not found under $vsInstall"
}

$cmd = "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && cl /std:c++17 /EHsc /O2 examples\cpp\native_relu.cpp /Fo:examples\cpp\native_relu.obj /Fe:examples\cpp\native_relu.exe"
cmd /c $cmd
