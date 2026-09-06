$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$stageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$targetRoot = 'D:\LaTex\Boundary Flow'
$beforeRoot = Join-Path $stageRoot 'before'
$assetSource = Join-Path $stageRoot 'Figures\HighResolution20260903'
$assetTarget = Join-Path $targetRoot 'Figures\HighResolution20260903'

$documentNames = @(
    'Methods Appendix.tex',
    'Methods Appendix.pdf',
    'PRL.tex',
    'PRL.pdf'
)

function Get-Sha256([string]$path) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file is missing: $path"
    }
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash
}

if (-not (Test-Path -LiteralPath $targetRoot -PathType Container)) {
    throw "Canonical publication directory is missing: $targetRoot"
}

# Refuse to overwrite a canonical document that changed after staging. An
# already-published staged hash is accepted so the script remains idempotent.
foreach ($name in $documentNames) {
    $canonical = Join-Path $targetRoot $name
    $before = Join-Path $beforeRoot $name
    $staged = Join-Path $stageRoot $name
    $canonicalHash = Get-Sha256 $canonical
    $beforeHash = Get-Sha256 $before
    $stagedHash = Get-Sha256 $staged
    if (($canonicalHash -ne $beforeHash) -and ($canonicalHash -ne $stagedHash)) {
        throw "Canonical file changed after staging; refusing to overwrite: $canonical"
    }
}

$pdfAssets = @(Get-ChildItem -LiteralPath $assetSource -File -Filter '*.pdf')
$pngAssets = @(Get-ChildItem -LiteralPath $assetSource -File -Filter '*.png')
if (($pdfAssets.Count -ne 13) -or ($pngAssets.Count -ne 13)) {
    throw "Expected 13 PDF and 13 PNG assets, found $($pdfAssets.Count) PDF and $($pngAssets.Count) PNG."
}

if (Test-Path -LiteralPath $assetTarget) {
    foreach ($source in @($pdfAssets + $pngAssets)) {
        $destination = Join-Path $assetTarget $source.Name
        if ((Test-Path -LiteralPath $destination -PathType Leaf) -and
            ((Get-Sha256 $destination) -ne (Get-Sha256 $source.FullName))) {
            throw "Existing publication asset differs; refusing to overwrite: $destination"
        }
    }
} else {
    New-Item -ItemType Directory -Path $assetTarget | Out-Null
}

# Publish assets first so the TeX sources never point at a missing directory.
foreach ($source in @($pdfAssets + $pngAssets)) {
    Copy-Item -LiteralPath $source.FullName -Destination (Join-Path $assetTarget $source.Name) -Force
}
foreach ($name in $documentNames) {
    Copy-Item -LiteralPath (Join-Path $stageRoot $name) -Destination (Join-Path $targetRoot $name) -Force
}

$results = @()
foreach ($name in $documentNames) {
    $source = Join-Path $stageRoot $name
    $destination = Join-Path $targetRoot $name
    $sourceHash = Get-Sha256 $source
    $destinationHash = Get-Sha256 $destination
    if ($sourceHash -ne $destinationHash) {
        throw "Post-publication hash mismatch: $destination"
    }
    $results += [pscustomobject]@{
        Kind = 'document'
        Name = $name
        Sha256 = $destinationHash
    }
}
foreach ($source in @($pdfAssets + $pngAssets)) {
    $destination = Join-Path $assetTarget $source.Name
    $sourceHash = Get-Sha256 $source.FullName
    $destinationHash = Get-Sha256 $destination
    if ($sourceHash -ne $destinationHash) {
        throw "Post-publication hash mismatch: $destination"
    }
    $results += [pscustomobject]@{
        Kind = 'figure'
        Name = $source.Name
        Sha256 = $destinationHash
    }
}

$results | Sort-Object Kind, Name | ConvertTo-Json -Depth 3
