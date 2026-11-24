param(
    [string] $WeightsDirectory = (Join-Path $PSScriptRoot '..\..\APAI4011_Week2_Weights')
)

$ErrorActionPreference = 'Stop'
$repo = 'xchlowex/Predicting-Partisanship-in-Congress'
$tag = 'week2-formal-weights-v1'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$weights = (Resolve-Path -LiteralPath $WeightsDirectory).Path
$manifests = @(Get-ChildItem (Join-Path $root 'results/week2/formal/*/run.json'))
if ($manifests.Count -ne 16) { throw 'Expected the 16 formal Week 2 run records.' }

$assets = foreach ($manifest in $manifests) {
    $record = Get-Content -Raw -LiteralPath $manifest.FullName | ConvertFrom-Json
    $path = Join-Path $weights ($manifest.Directory.Name + '.pt')
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash -ne $record.checkpoint_sha256) {
        throw "Wrong checkpoint for $($manifest.Directory.Name)"
    }
    $path
}

$checksums = Join-Path $root 'results/week2/formal_summary/checkpoint_sha256.txt'
$target = gh api "repos/$repo/git/ref/heads/main" --jq '.object.sha'
if ($LASTEXITCODE -ne 0) { throw "Sign in to GitHub CLI with write access to $repo." }
$notes = '16 development-selected Week 2 checkpoints, each named for its formal run. See reports/week3_weights.md on main for installation, checksums, and edited-speech inference.'
gh release create $tag @assets $checksums --repo $repo --target $target --title 'Week 2 formal model weights' --notes $notes
if ($LASTEXITCODE -ne 0) { throw 'Release creation failed. Inspect the release state before retrying.' }
