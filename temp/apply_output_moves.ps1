$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path.TrimEnd('\')
$outputRoot = (Resolve-Path -LiteralPath (Join-Path $workspaceRoot 'output')).Path.TrimEnd('\')
if ($outputRoot -ne [IO.Path]::Combine($workspaceRoot, 'output')) { throw 'Unexpected output root' }
$outputPrefix = $outputRoot + '\'
$plan = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'output_move_plan.json') -Raw | ConvertFrom-Json
$records = [Collections.Generic.List[object]]::new()
function Assert-OutputPath([string] $candidate) {
    $absolute = [IO.Path]::GetFullPath($candidate)
    if (-not $absolute.StartsWith($outputPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path outside output: $absolute"
    }
    $ancestor = $absolute
    while ($ancestor -and $ancestor.Length -ge $outputRoot.Length) {
        if (Test-Path -LiteralPath $ancestor) {
            if ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Linked path requires manual review: $ancestor"
            }
        }
        $ancestor = [IO.Path]::GetDirectoryName($ancestor)
    }
}
function Get-Inventory([string] $target) {
    $info = Get-Item -LiteralPath $target -Force
    $items = if ($info.PSIsContainer) { @(Get-ChildItem -LiteralPath $target -Recurse -Force) } else { @($info) }
    if (@($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
        throw "Linked contents require manual review: $target"
    }
    @($items | ForEach-Object {
        $relative = if ($info.PSIsContainer) { $_.FullName.Substring($target.Length) } else { '' }
        if ($_.PSIsContainer) { 'D|' + $relative }
        else { 'F|' + $relative + '|' + $_.Length + '|' + $_.LastWriteTimeUtc.Ticks }
    } | Sort-Object)
}
foreach ($item in $plan) {
    $source = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $item.source))
    $destination = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $item.destination))
    Assert-OutputPath $source
    Assert-OutputPath $destination
    if (-not (Test-Path -LiteralPath $source)) { throw "Source missing: $source" }
    if (Test-Path -LiteralPath $destination) { throw "Destination already exists: $destination" }
    $records.Add([pscustomobject]@{ Source = $item.source; Destination = $item.destination; Status = 'Planned'; EntriesVerified = 0 })
}
$reportDir = Join-Path $outputRoot '_organization'
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
$manifest = Join-Path $reportDir 'moves_2026-10-16.csv'
if (Test-Path -LiteralPath $manifest) { throw 'Migration manifest already exists' }
$records | Export-Csv -LiteralPath $manifest -NoTypeInformation -Encoding UTF8
$index = 0
foreach ($record in $records) {
    $source = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $record.Source))
    $destination = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $record.Destination))
    Assert-OutputPath $source
    Assert-OutputPath $destination
    $before = @(Get-Inventory $source)
    New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($destination)) -Force | Out-Null
    Move-Item -LiteralPath $source -Destination $destination -ErrorAction Stop
    $record.Status = 'Moved'
    $records | Export-Csv -LiteralPath $manifest -NoTypeInformation -Encoding UTF8
    $after = @(Get-Inventory $destination)
    if ($before.Count -ne $after.Count -or ($before.Count -gt 0 -and @(Compare-Object $before $after).Count)) {
        throw "Inventory verification failed: $destination; see $manifest"
    }
    $record.Status = 'Verified'
    $record.EntriesVerified = $after.Count
    $records | Export-Csv -LiteralPath $manifest -NoTypeInformation -Encoding UTF8
    $index++
    if ($index % 50 -eq 0) { Write-Output "Verified $index of $($records.Count) moves" }
}
Write-Output "All $index moves verified (names, sizes, timestamps, and nested directories). Manifest: $manifest"
