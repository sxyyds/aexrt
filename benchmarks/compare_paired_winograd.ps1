param(
    [ValidateRange(1, 1000000)]
    [int]$Runs = 1000,
    [ValidateRange(1, 1000)]
    [int]$Rounds = 5,
    [string]$BaselineEngine = "build\native\paired_winograd_engines\paired_winograd_baseline.aexrt",
    [string]$CandidateEngine = "build\native\paired_winograd_engines\paired_winograd_candidate.aexrt"
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

function Get-SingleProfileMs([string]$Text, [string]$Pattern, [string]$Description) {
    $profileMatches = [regex]::Matches($Text, $Pattern)
    if ($profileMatches.Count -ne 1) {
        throw "expected one $Description profile event, found $($profileMatches.Count):`n$Text"
    }
    return Convert-InvariantDouble $profileMatches[0].Groups[1].Value
}

function Invoke-PairedBenchmark([int]$Round, [string]$Name, [string]$Engine) {
    $output = @(& $executable $Engine --runs $Runs --profile 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code ${LASTEXITCODE}:`n$($output -join "`n")"
    }
    $text = $output -join "`n"
    $inferMatch = [regex]::Match($text, "avg_infer=([0-9.]+) ms")
    if (-not $inferMatch.Success) {
        throw "could not parse average inference time for ${Name}:`n$text"
    }

    if ($Name -eq "baseline") {
        $cmd0Ms = Get-SingleProfileMs $text "(?m)^\s*\[\s*\d+\]\s+([0-9.]+) ms\s+cmd#0 CONV_SILU\b" "baseline cmd#0 CONV_SILU"
        $cmd1Ms = Get-SingleProfileMs $text "(?m)^\s*\[\s*\d+\]\s+([0-9.]+) ms\s+cmd#1 CONV_SILU\b" "baseline cmd#1 CONV_SILU"
        $gpuPairMs = $cmd0Ms + $cmd1Ms
        $event = "cmd#0 + cmd#1"
    } else {
        $gpuPairMs = Get-SingleProfileMs $text "(?m)^\s*\[\s*\d+\]\s+([0-9.]+) ms\s+cmd#0 PAIRED_CONV3X3_SILU\b" "candidate PAIRED_CONV3X3_SILU"
        $event = "PAIRED_CONV3X3_SILU"
    }

    [pscustomobject]@{
        Round = $Round
        Case = $Name
        CpuAvgMs = Convert-InvariantDouble $inferMatch.Groups[1].Value
        GpuPairMs = $gpuPairMs
        ProfileEvent = $event
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
    foreach ($path in @($baselinePath, $candidatePath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "benchmark engine not found: $path"
        }
    }

    $results = @()
    for ($round = 1; $round -le $Rounds; ++$round) {
        $cases = if (($round % 2) -eq 1) {
            @(@("baseline", $baselinePath), @("candidate", $candidatePath))
        } else {
            @(@("candidate", $candidatePath), @("baseline", $baselinePath))
        }
        foreach ($case in $cases) {
            $results += Invoke-PairedBenchmark $round $case[0] $case[1]
        }
    }

    $results | Format-Table Round, Case, CpuAvgMs, GpuPairMs, ProfileEvent -AutoSize
    $medians = foreach ($group in ($results | Group-Object Case | Sort-Object Name)) {
        [pscustomobject]@{
            Case = $group.Name
            CpuMedianMs = Get-Median @($group.Group.CpuAvgMs)
            GpuPairMedianMs = Get-Median @($group.Group.GpuPairMs)
        }
    }
    $medians | Format-Table Case, CpuMedianMs, GpuPairMedianMs -AutoSize

    $baselineMedian = ($medians | Where-Object Case -eq "baseline").GpuPairMedianMs
    $candidateMedian = ($medians | Where-Object Case -eq "candidate").GpuPairMedianMs
    $deltaMs = $candidateMedian - $baselineMedian
    $changePercent = if ($baselineMedian -gt 0.0) { 100.0 * $deltaMs / $baselineMedian } else { 0.0 }
    Write-Host ("paired GPU median: baseline={0:F4} ms candidate={1:F4} ms delta={2:+0.0000;-0.0000;0.0000} ms ({3:+0.0;-0.0;0.0}%)" -f $baselineMedian, $candidateMedian, $deltaMs, $changePercent)
} finally {
    Pop-Location
}
