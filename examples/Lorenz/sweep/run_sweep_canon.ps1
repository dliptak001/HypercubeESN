# Canonical A-vs-tanh comparison on the LOCKED general-purpose seeds (M=16).
# Seeds 23, 42, 73895 serve both activations (ranked by worst-of-two-arm VPT).
# NOTE: no $ErrorActionPreference='Stop' — PS 5.1 turns native-exe stderr into a
# terminating error that would kill the loop mid-sweep.
$ErrorActionPreference = 'Continue'
$env:PATH = "C:\Program Files\JetBrains\CLion 2024.3.2\bin\mingw\bin;" + $env:PATH
$exe = "C:\CLion\HypercubeESN\cmake-build-release\Lorenz.exe"
$dir = "C:\CLion\HypercubeESN\examples\Lorenz\sweep\canon"
New-Item -ItemType Directory -Force $dir | Out-Null

# history_depth pinned to 16 in source; arms differ only by gamma (0=tanh, 1.1=A).
$lockedSeeds = 23, 42, 73895
$combos = New-Object System.Collections.ArrayList
foreach ($s in $lockedSeeds) {
    [void]$combos.Add(@($s, '0'))
    [void]$combos.Add(@($s, '1.1'))
}

$i = 0
foreach ($c in $combos) {
    $i++
    $seed = $c[0]; $g = $c[1]
    $tag = "s${seed}_g${g}"
    Write-Host "[$i/$($combos.Count)] running $tag ..."
    & $exe 0.88 0.10 $seed $g | Out-File -Encoding utf8 "$dir\$tag.txt"
}
Write-Host "SWEEP DONE: $($combos.Count) runs in $dir"
