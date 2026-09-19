param(
    [ValidateRange(1, 1000000)]
    [int]$Runs = 1000,
    [ValidateRange(1, 1000)]
    [int]$Rounds = 5,
    [string]$BaselineEngine = "build\native\hotspot_engines\conv40_64x64_alg39.aexrt",
    [string]$CandidateEngine = "build\native\hotspot_engines\conv40_64x64_alg44.aexrt"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$executable = ".\examples\cpp\bin\native_yolo_package.exe"

function Convert-ToWorkspaceArgument([string]$Path) {
    if (-not [System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    $prefix = $root.TrimEnd("\") + "\"
    if ($Path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        return ".\" + $Path.Substring($prefix.Length)
    }
    return $Path
}

function Convert-InvariantDouble([string]$Value) {
    return [double]::Parse($Value, [Globalization.CultureInfo]::InvariantCulture)
}

function Invoke-HotspotBenchmark([int]$Round, [string]$Name, [string]$Engine) {
    $output = @(& $executable $Engine --runs $Runs --profile 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code ${LASTEXITCODE}:`n$($output -join "`n")"
    }
    $text = $output -join "`n"
    $inferMatch = [regex]::Match($text, "avg_infer=([0-9.]+) ms")
    $convMatch = [regex]::Match(
        $text,
        "(?m)^\s*\[\s*0\]\s+([0-9.]+) ms\s+cmd#0 CONV_SILU\b.*\balg=([^\s]+)"
    )
    if (-not $inferMatch.Success -or -not $convMatch.Success) {
        throw "could not parse benchmark output for ${Name}:`n$text"
    }
    [pscustomobject]@{
        Round = $Round
        Kernel = $Name
        Algorithm = $convMatch.Groups[2].Value
        CpuAvgMs = Convert-InvariantDouble $inferMatch.Groups[1].Value
        GpuConvMs = Convert-InvariantDouble $convMatch.Groups[1].Value
    }
}

function Get-Median([double[]]$Values) {
    $sorted = @($Values | Sort-Object)
    $middle = [int][Math]::Floor($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 1) {
        return $sorted[$middle]
    }
    return ($sorted[$middle - 1] + $sorted[$middle]) / 2.0
}

Push-Location $root
try {
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "benchmark executable not found: $executable"
    }
    $baselinePath = Convert-ToWorkspaceArgument $BaselineEngine
    $candidatePath = Convert-ToWorkspaceArgument $CandidateEngine
    if (-not (Test-Path -LiteralPath $baselinePath -PathType Leaf)) {
        throw "baseline engine not found: $baselinePath"
    }
    if (-not (Test-Path -LiteralPath $candidatePath -PathType Leaf)) {
        throw "candidate engine not found: $candidatePath"
    }
    $results = @()
    for ($round = 1; $round -le $Rounds; ++$round) {
        $cases = if (($round % 2) -eq 1) {
            @(@("ID39", $baselinePath), @("ID44", $candidatePath))
        } else {
            @(@("ID44", $candidatePath), @("ID39", $baselinePath))
        }
        foreach ($case in $cases) {
            $results += Invoke-HotspotBenchmark $round $case[0] $case[1]
        }
    }

    $results | Format-Table Round, Kernel, Algorithm, CpuAvgMs, GpuConvMs -AutoSize
    $medians = foreach ($group in ($results | Group-Object Kernel | Sort-Object Name)) {
        [pscustomobject]@{
            Kernel = $group.Name
            CpuMedianMs = Get-Median @($group.Group.CpuAvgMs)
            GpuConvMedianMs = Get-Median @($group.Group.GpuConvMs)
        }
    }
    $medians | Format-Table Kernel, CpuMedianMs, GpuConvMedianMs -AutoSize
} finally {
    Pop-Location
}
