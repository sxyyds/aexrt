param(
    [ValidateRange(1, 1000000)] [int]$Runs = 1000,
    [ValidateRange(1, 1000)] [int]$Rounds = 5,
    [string]$BaselineEngine = "build\native\cs2V8_320_no_pair.aexrt",
    [string]$CandidateEngine = "examples\cs2V8_320.aexrt"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$executable = ".\examples\cpp\bin\native_yolo_package.exe"

function Parse-Double([string]$Value) {
    return [double]::Parse($Value, [Globalization.CultureInfo]::InvariantCulture)
}

function Match-One([string]$Text, [string]$Pattern, [string]$Description) {
    $matches = [regex]::Matches($Text, $Pattern)
    if ($matches.Count -ne 1) {
        throw "expected one $Description, found $($matches.Count):`n$Text"
    }
    return Parse-Double $matches[0].Groups[1].Value
}

function Invoke-Case([int]$Round, [string]$Name, [string]$Engine) {
    $output = @(& $executable $Engine --runs $Runs --profile 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code ${LASTEXITCODE}:`n$($output -join "`n")"
    }
    $text = $output -join "`n"
    $cpu = Match-One $text "avg_infer=([0-9.]+) ms" "average inference"
    $gpu = Match-One $text "summed_gpu_ms=([0-9.]+)" "GPU dispatch sum"
    if ($Name -eq "baseline") {
        $pair = (Match-One $text "(?m)^\s*\[\s*\d+\]\s+([0-9.]+) ms\s+cmd#90 CONV_SILU\b" "cmd#90") +
            (Match-One $text "(?m)^\s*\[\s*\d+\]\s+([0-9.]+) ms\s+cmd#91 CONV_SILU\b" "cmd#91")
    } else {
        $pair = Match-One $text "(?m)^\s*\[\s*\d+\]\s+([0-9.]+) ms\s+cmd#90 PAIRED_CONV3X3_SILU\b" "paired cmd#90"
    }
    [pscustomobject]@{ Round = $Round; Case = $Name; CpuMs = $cpu; GpuMs = $gpu; PairMs = $pair }
}

function Median([double[]]$Values) {
    $sorted = @($Values | Sort-Object)
    $middle = [int][Math]::Floor($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 1) { return $sorted[$middle] }
    return ($sorted[$middle - 1] + $sorted[$middle]) / 2.0
}

Push-Location $root
try {
    $results = @()
    for ($round = 1; $round -le $Rounds; ++$round) {
        $cases = if (($round % 2) -eq 1) {
            @(@("baseline", $BaselineEngine), @("candidate", $CandidateEngine))
        } else {
            @(@("candidate", $CandidateEngine), @("baseline", $BaselineEngine))
        }
        foreach ($case in $cases) { $results += Invoke-Case $round $case[0] $case[1] }
    }
    $results | Format-Table Round, Case, CpuMs, GpuMs, PairMs -AutoSize
    $medians = foreach ($group in ($results | Group-Object Case | Sort-Object Name)) {
        [pscustomobject]@{
            Case = $group.Name
            CpuMedianMs = Median @($group.Group.CpuMs)
            GpuMedianMs = Median @($group.Group.GpuMs)
            PairMedianMs = Median @($group.Group.PairMs)
        }
    }
    $medians | Format-Table Case, CpuMedianMs, GpuMedianMs, PairMedianMs -AutoSize
} finally {
    Pop-Location
}
