[CmdletBinding()]
param(
  [Parameter(Mandatory = $false)]
  [string]$DestinationDir = $(Join-Path $PSScriptRoot 'katago-v1.16.4-openvino2026.0-npu-windows-x64'),

  [Parameter(Mandatory = $false)]
  [string[]]$RequiredDllNames = @(
    'abseil_dll.dll',
    'libprotobuf-lite.dll',
    'libprotobuf.dll',
    'msvcp140.dll',
    'msvcp140_1.dll',
    'msvcp140_2.dll',
    'msvcp140_atomic_wait.dll',
    'msvcp140_codecvt_ids.dll',
    'onnxruntime.dll',
    'onnxruntime_providers_openvino.dll',
    'onnxruntime_providers_shared.dll',
    'openvino.dll',
    'openvino_intel_npu_plugin.dll',
    'openvino_onnx_frontend.dll',
    'tbb12.dll',
    'vcruntime140.dll',
    'vcruntime140_1.dll',
    'zlib1.dll'
  )
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-FullPath {
  param([Parameter(Mandatory = $true)][string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return [System.IO.Path]::GetFullPath($Path)
  }

  return [System.IO.Path]::GetFullPath((Join-Path (Get-Location).Path $Path))
}

function Get-LatestMsvcCrtDir {
  param([Parameter(Mandatory = $true)][string]$MsvcRedistRoot)

  if (-not (Test-Path -LiteralPath $MsvcRedistRoot -PathType Container)) {
    throw "MSVC redist root not found: $MsvcRedistRoot"
  }

  $versionDirs = Get-ChildItem -LiteralPath $MsvcRedistRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    try {
      [PSCustomObject]@{
        Name    = $_.Name
        Full    = $_.FullName
        Version = [Version]$_.Name
      }
    }
    catch {
      # Ignore non-version folders.
    }
  }

  if ($versionDirs.Count -eq 0) {
    throw "No MSVC version folder found under: $MsvcRedistRoot"
  }

  $ordered = $versionDirs | Sort-Object -Property @{ Expression = 'Version'; Descending = $true }

  foreach ($entry in $ordered) {
    $x64Dir = Join-Path $entry.Full 'x64'
    if (-not (Test-Path -LiteralPath $x64Dir -PathType Container)) {
      continue
    }

    $crtDirs = Get-ChildItem -LiteralPath $x64Dir -Directory -ErrorAction SilentlyContinue | Where-Object {
      $_.Name -match '^Microsoft\.VC\d+\.CRT$'
    } | ForEach-Object {
      $m = [regex]::Match($_.Name, '^Microsoft\.VC(\d+)\.CRT$')
      [PSCustomObject]@{
        Path     = $_.FullName
        VcNumber = [int]$m.Groups[1].Value
      }
    }

    if ($crtDirs.Count -gt 0) {
      $bestCrt = $crtDirs |
        Sort-Object -Property @(
          @{ Expression = 'VcNumber'; Descending = $true },
          @{ Expression = 'Path'; Descending = $false }
        ) |
        Select-Object -First 1

      return $bestCrt.Path
    }
  }

  throw "Cannot find Microsoft.VC*.CRT under: $MsvcRedistRoot"
}

function Get-OpenVinoRoots {
  param([Parameter(Mandatory = $true)][string]$ProgramFilesX86)

  $roots = New-Object System.Collections.Generic.List[string]

  $releasePattern = Join-Path $ProgramFilesX86 'Intel\openvino_*\runtime\bin\intel64\Release'
  Get-ChildItem -Path $releasePattern -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    [void]$roots.Add($_.FullName)
  }

  $tbbRoot = Join-Path $ProgramFilesX86 'Intel\openvino_*\runtime\3rdparty\tbb\bin'
  if (Test-Path -LiteralPath $tbbRoot -PathType Container) {
    [void]$roots.Add((Resolve-Path -LiteralPath $tbbRoot).Path)
  }

  return $roots | Sort-Object -Unique
}

function Get-DllGroup {
  param([Parameter(Mandatory = $true)][string]$DllName)

  $n = $DllName.ToLowerInvariant()

  if ($n.StartsWith('onnxruntime')) {
    return 'onnxruntime'
  }

  if ($n.StartsWith('openvino') -or $n -eq 'tbb12.dll') {
    return 'openvino'
  }

  if ($n -match '^(msvcp|vcruntime|concrt|vccorlib).+\.dll$') {
    return 'msvc'
  }

  return 'deps'
}

function Find-DllCandidates {
  param(
    [Parameter(Mandatory = $true)][string]$DllName,
    [Parameter(Mandatory = $true)][string[]]$Roots
  )

  $hits = New-Object System.Collections.Generic.List[string]
  foreach ($root in $Roots) {
    if (-not (Test-Path -LiteralPath $root -PathType Container)) {
      continue
    }

    Get-ChildItem -LiteralPath $root -Filter $DllName -File -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
      if (-not $hits.Contains($_.FullName)) {
        [void]$hits.Add($_.FullName)
      }
    }
  }

  return $hits.ToArray()
}

function Select-BestCandidate {
  param([Parameter(Mandatory = $true)][string[]]$Paths)

  $best = $Paths | ForEach-Object {
    try {
      $item = Get-Item -LiteralPath $_ -ErrorAction Stop
      [PSCustomObject]@{
        Path      = $_
        LastWrite = $item.LastWriteTimeUtc
      }
    }
    catch {
      # Skip unreadable candidates.
    }
  } |
  Sort-Object -Property @(
    @{ Expression = 'LastWrite'; Descending = $true },
    @{ Expression = 'Path'; Descending = $false }
  ) |
  Select-Object -First 1

  if ($null -eq $best) {
    return $null
  }

  return $best.Path
}

$repoRoot = Resolve-FullPath -Path (Join-Path $PSScriptRoot '..')
$destinationAbs = Resolve-FullPath -Path $DestinationDir

New-Item -ItemType Directory -Path $destinationAbs -Force | Out-Null

$programFilesX86 = ${env:ProgramFiles(x86)}
if ([string]::IsNullOrWhiteSpace($programFilesX86)) {
  throw 'Environment variable ProgramFiles(x86) is empty.'
}

$onnxRuntimeRoot = Join-Path $repoRoot 'cpp\build\Release'
$depsRoot = Join-Path $repoRoot 'cpp\build\deps'
$msvcRedistRoot = Join-Path $programFilesX86 'Microsoft Visual Studio\18\BuildTools\VC\Redist\MSVC'

$openvinoRoots = Get-OpenVinoRoots -ProgramFilesX86 $programFilesX86
$msvcCrtRoot = Get-LatestMsvcCrtDir -MsvcRedistRoot $msvcRedistRoot

$groupRoots = @{
  openvino   = $openvinoRoots
  msvc       = @($msvcCrtRoot)
  onnxruntime = @($onnxRuntimeRoot)
  deps       = @($depsRoot)
}

Write-Host "Destination: $destinationAbs"
Write-Host "OpenVINO roots: $($openvinoRoots.Count)"
$openvinoRoots | ForEach-Object { Write-Host "  - $_" }
Write-Host "MSVC CRT root: $msvcCrtRoot"
Write-Host "ONNX Runtime root: $onnxRuntimeRoot"
Write-Host "Deps root: $depsRoot"
Write-Host "Required DLL count: $($RequiredDllNames.Count)"

$copied = New-Object System.Collections.Generic.List[object]
$missing = New-Object System.Collections.Generic.List[string]

foreach ($dllName in ($RequiredDllNames | Sort-Object -Unique)) {
  $group = Get-DllGroup -DllName $dllName
  $roots = $groupRoots[$group]

  if ($null -eq $roots -or $roots.Count -eq 0) {
    [void]$missing.Add($dllName)
    Write-Warning "No search root for $dllName (group=$group)"
    continue
  }

  Write-Host "[Find] $dllName (group=$group)"
  $candidates = Find-DllCandidates -DllName $dllName -Roots $roots
  if ($candidates.Count -eq 0) {
    [void]$missing.Add($dllName)
    Write-Warning "Missing: $dllName"
    continue
  }

  $best = Select-BestCandidate -Paths $candidates
  if ($null -eq $best) {
    [void]$missing.Add($dllName)
    Write-Warning "Unreadable candidates: $dllName"
    continue
  }

  $target = Join-Path $destinationAbs $dllName
  Copy-Item -LiteralPath $best -Destination $target -Force

  [void]$copied.Add([PSCustomObject]@{
    Dll            = $dllName
    Group          = $group
    Source         = $best
    Destination    = $target
    CandidateCount = $candidates.Count
  })

  Write-Host "[Copy] $dllName <= $best"
}

$reportPath = Join-Path $destinationAbs 'dll-copy-report.txt'
$reportLines = New-Object System.Collections.Generic.List[string]
$reportLines.Add("DestinationDir: $destinationAbs") | Out-Null
$reportLines.Add("OpenVINO roots: $($openvinoRoots.Count)") | Out-Null
$openvinoRoots | ForEach-Object { $reportLines.Add("  - $_") | Out-Null }
$reportLines.Add("MSVC CRT root: $msvcCrtRoot") | Out-Null
$reportLines.Add("ONNX Runtime root: $onnxRuntimeRoot") | Out-Null
$reportLines.Add("Deps root: $depsRoot") | Out-Null
$reportLines.Add("Copied: $($copied.Count)") | Out-Null
$reportLines.Add("Missing: $($missing.Count)") | Out-Null
$reportLines.Add('') | Out-Null
$reportLines.Add('[Copied]') | Out-Null

$copied |
  Sort-Object -Property Dll |
  ForEach-Object {
    $reportLines.Add("$($_.Dll)`tGroup=$($_.Group)`t$($_.Source)`tCandidates=$($_.CandidateCount)") | Out-Null
  }

$reportLines.Add('') | Out-Null
$reportLines.Add('[Missing]') | Out-Null
$missing |
  Sort-Object |
  ForEach-Object {
    $reportLines.Add($_) | Out-Null
  }

Set-Content -LiteralPath $reportPath -Value $reportLines -Encoding UTF8

Write-Host ''
Write-Host "Done. Copied $($copied.Count)/$($RequiredDllNames.Count) DLLs"
Write-Host "Output: $destinationAbs"
Write-Host "Report: $reportPath"

if ($missing.Count -gt 0) {
  exit 2
}
