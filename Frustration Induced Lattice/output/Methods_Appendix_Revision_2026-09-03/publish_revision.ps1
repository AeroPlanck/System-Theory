$ErrorActionPreference = 'Stop'
$revisionRoot = 'D:\PythonProject\System Theory\Frustration Induced Lattice\output\Methods_Appendix_Revision_2026-09-03'
$latexRoot = 'D:\LaTex\Boundary Flow'
$theoryRoot = 'D:\PrivatePythonProject\Math\Lattice'

# Validate every existing source against its preserved pre-revision snapshot
# before making any external write. An already-published identical file is safe.
$sourcePairs = @(
    @('Methods Appendix.tex', $latexRoot, ''),
    @('PRL.tex', $latexRoot, ''),
    @('ChernNumberCompute.py', $theoryRoot, 'code'),
    @('ChernParamScan.py', $theoryRoot, 'code'),
    @('Dispersion.py', $theoryRoot, 'code'),
    @('SpectralFlow.py', $theoryRoot, 'code')
)
foreach ($pair in $sourcePairs) {
    $snapshot = Join-Path (Join-Path $revisionRoot 'before') $pair[0]
    $staged = Join-Path (Join-Path $revisionRoot $pair[2]) $pair[0]
    $destination = Join-Path $pair[1] $pair[0]
    $currentHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
    $beforeHash = (Get-FileHash -LiteralPath $snapshot -Algorithm SHA256).Hash
    $newHash = (Get-FileHash -LiteralPath $staged -Algorithm SHA256).Hash
    if ($currentHash -ne $beforeHash -and $currentHash -ne $newHash) {
        throw "Concurrent source change detected: $destination"
    }
}

$newPairs = @(
    @('code\run_strip_matched_sweep.py', (Join-Path $theoryRoot 'run_strip_matched_sweep.py')),
    @('figures\strip_circle_matched_alpha_sweep.pdf', (Join-Path $latexRoot 'Figures\strip_circle_matched_alpha_sweep.pdf')),
    @('figures\strip_circle_matched_alpha_sweep.png', (Join-Path $latexRoot 'Figures\strip_circle_matched_alpha_sweep.png')),
    @('figures\strip_circle_matched_alpha_050.png', (Join-Path $latexRoot 'Figures\strip_circle_matched_alpha_050.png')),
    @('figures\single_v_3_omega_0_gap_screened.png', (Join-Path $latexRoot 'Figures\single_v_3_omega_0_gap_screened.png')),
    @('figures\mixed_v_3_omega_0_gap_screened.png', (Join-Path $latexRoot 'Figures\mixed_v_3_omega_0_gap_screened.png'))
)
foreach ($pair in $newPairs) {
    $staged = Join-Path $revisionRoot $pair[0]
    if (-not (Test-Path -LiteralPath $staged -PathType Leaf)) { throw "Missing artifact: $staged" }
    if (Test-Path -LiteralPath $pair[1]) {
        if ((Get-FileHash -LiteralPath $pair[1]).Hash -ne (Get-FileHash -LiteralPath $staged).Hash) {
            throw "Refusing to overwrite a different new-name artifact: $($pair[1])"
        }
    }
}

$pdfSource = Join-Path $revisionRoot 'Methods Appendix.pdf'
if (-not (Test-Path -LiteralPath $pdfSource -PathType Leaf)) { throw 'Compiled appendix PDF missing' }
$pdfDestination = Join-Path $latexRoot 'Methods Appendix.pdf'
$pdfBackup = Join-Path $revisionRoot 'before\Methods Appendix.pdf'
if ((Test-Path -LiteralPath $pdfDestination) -and -not (Test-Path -LiteralPath $pdfBackup)) {
    Copy-Item -LiteralPath $pdfDestination -Destination $pdfBackup
}

foreach ($pair in $sourcePairs) {
    $staged = Join-Path (Join-Path $revisionRoot $pair[2]) $pair[0]
    Copy-Item -LiteralPath $staged -Destination (Join-Path $pair[1] $pair[0])
}
foreach ($pair in $newPairs) {
    Copy-Item -LiteralPath (Join-Path $revisionRoot $pair[0]) -Destination $pair[1]
}
Copy-Item -LiteralPath $pdfSource -Destination $pdfDestination

foreach ($pair in $sourcePairs) {
    $staged = Join-Path (Join-Path $revisionRoot $pair[2]) $pair[0]
    $destination = Join-Path $pair[1] $pair[0]
    if ((Get-FileHash -LiteralPath $staged).Hash -ne (Get-FileHash -LiteralPath $destination).Hash) {
        throw "Published hash mismatch: $destination"
    }
}
Write-Output 'Published six verified sources, the strip runner, five new-name figure assets, and the appendix PDF.'
Write-Output 'Original sources/PDF remain in before; previous figure assets were not overwritten.'
