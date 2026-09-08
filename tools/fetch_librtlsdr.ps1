<#
.SYNOPSIS
    Download the official rtl-sdr-blog Windows x64 build into lib/.

.DESCRIPTION
    64-bit Python cannot load the 32-bit rtlsdr.dll that ships inside the
    x86 SDR# download, so this fetches the x64 build and vendors the three
    files the app needs:

        rtlsdr.dll        the driver library pyrtlsdr binds to
        libusb-1.0.dll    its USB dependency
        rtl_test.exe      standalone sanity check for the dongle

    Nothing is installed system-wide and nothing is added to PATH; the app
    adds lib/ to its own DLL search path at startup.

.PARAMETER Url
    Override the release archive URL.

.PARAMETER Force
    Overwrite files already present in lib/.
#>
[CmdletBinding()]
param(
    [string]$Url = "https://github.com/rtlsdrblog/rtl-sdr-blog/releases/download/1.3.5/Release.zip",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$root   = Split-Path -Parent $PSScriptRoot
$libDir = Join-Path $root "lib"
$want   = @("rtlsdr.dll", "libusb-1.0.dll", "rtl_test.exe")

if (-not (Test-Path $libDir)) { New-Item -ItemType Directory -Path $libDir | Out-Null }

$existing = $want | Where-Object { Test-Path (Join-Path $libDir $_) }
if ($existing -and -not $Force) {
    Write-Host "Already present in lib/: $($existing -join ', ')" -ForegroundColor Yellow
    Write-Host "Re-run with -Force to overwrite."
    exit 0
}

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("librtlsdr_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp | Out-Null
$zip = Join-Path $tmp "release.zip"

try {
    Write-Host "Downloading $Url" -ForegroundColor Cyan
    Invoke-WebRequest -Uri $Url -OutFile $zip -UseBasicParsing

    $size = (Get-Item $zip).Length
    $hash = (Get-FileHash $zip -Algorithm SHA256).Hash
    Write-Host ("  {0:N0} bytes" -f $size)
    Write-Host "  SHA256 $hash"

    Write-Host "Extracting..." -ForegroundColor Cyan
    Expand-Archive -Path $zip -DestinationPath $tmp -Force

    $copied = @()
    foreach ($name in $want) {
        # Prefer a path with x64 in it; the archive carries both builds.
        $found = Get-ChildItem -Path $tmp -Filter $name -Recurse -File |
                 Sort-Object @{ Expression = { $_.FullName -match 'x64|64bit' } } -Descending |
                 Select-Object -First 1
        if (-not $found) {
            Write-Warning "$name not found in the archive"
            continue
        }
        Copy-Item $found.FullName -Destination (Join-Path $libDir $name) -Force
        $copied += $name
        Write-Host "  $name  <- $($found.FullName.Substring($tmp.Length+1))"
    }

    if ($copied -notcontains "rtlsdr.dll") {
        throw "rtlsdr.dll was not found in the archive. Check the URL."
    }

    # Verify we actually got the 64-bit build by reading the PE header,
    # rather than trusting the folder name.
    $dll = Join-Path $libDir "rtlsdr.dll"
    $fs = [System.IO.File]::OpenRead($dll)
    try {
        $br = New-Object System.IO.BinaryReader($fs)
        $fs.Position = 0x3C
        $peOff = $br.ReadInt32()
        $fs.Position = $peOff + 4
        $machine = $br.ReadUInt16()
    } finally { $fs.Dispose() }

    switch ($machine) {
        0x8664 { Write-Host "`nrtlsdr.dll verified as x64. Ready." -ForegroundColor Green }
        0x14c  { throw "Downloaded rtlsdr.dll is 32-bit (x86). 64-bit Python cannot load it." }
        default { throw ("Unrecognised machine type 0x{0:X4} in rtlsdr.dll" -f $machine) }
    }

    Write-Host "`nNext: .venv\Scripts\python.exe tools\check_device.py" -ForegroundColor Cyan
}
finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
