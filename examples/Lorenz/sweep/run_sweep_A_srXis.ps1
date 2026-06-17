# A(x)-only SR x IS grid at the LOCKED seeds (M=16, leak=1.0).
#   activation : A_lorentz, gamma=1.1, inv_sigma2=250 (defaults)
#   SR grid    : 0.85, 0.875, 0.90
#   IS grid    : 0.20, 0.10, 0.05
#   seeds      : 23, 42, 73895   => 3 x 3 x 3 = 27 runs (~72 min)
# leak_rate left at source default 1.0 (argv[6] omitted).
# NOTE: no $ErrorActionPreference='Stop' — PS 5.1 turns native-exe stderr into a
# terminating error that would kill the loop mid-sweep.
$ErrorActionPreference = 'Continue'
$env:PATH = "C:\Program Files\JetBrains\CLion 2024.3.2\bin\mingw\bin;" + $env:PATH
$exe = "C:\CLion\HypercubeESN\cmake-build-release\Lorenz.exe"
$dir = "C:\CLion\HypercubeESN\examples\Lorenz\sweep\A_srXis_d16"
New-Item -ItemType Directory -Force $dir | Out-Null

$seeds = 23, 42, 73895
$srs   = '0.85', '0.875', '0.90'
$iss   = '0.20', '0.10', '0.05'
$gamma = '1.1'   # A(x)
$isig  = '250'   # inv_sigma2 default

$combos = New-Object System.Collections.ArrayList
foreach ($s in $seeds) {
    foreach ($sr in $srs) {
        foreach ($is in $iss) {
            [void]$combos.Add(@($s, $sr, $is))
        }
    }
}

$i = 0
foreach ($c in $combos) {
    $i++
    $seed = $c[0]; $sr = $c[1]; $is = $c[2]
    $tag = "s${seed}_sr${sr}_is${is}"
    Write-Host "[$i/$($combos.Count)] running $tag ..."
    & $exe $sr $is $seed $gamma $isig | Out-File -Encoding utf8 "$dir\$tag.txt"
}
Write-Host "SWEEP DONE: $($combos.Count) runs in $dir"
