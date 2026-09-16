# Installs Wavefield from source - the terminal alternative to the .exe
# installer.
#
# For when Windows Defender/Smart App Control/an IT policy won't let the
# installer run (see README.md). This downloads Wavefield's source from
# GitHub and its dependencies from PyPI, exactly like a developer setting the
# project up by hand per docs/DEVELOPERS.md, but automatically: it also
# creates a private virtual environment (so it cannot collide with anything
# else on the machine), sets up WhisperX the same way the installer's
# post-install step does, and adds a Start Menu shortcut - so what comes out
# the other end is as usable as an installed copy, without ever running an
# executable someone else built.
#
# Usage, in PowerShell:
#   irm https://raw.githubusercontent.com/pcmflores19-ph/wavefield/main/packaging/install-wavefield.ps1 | iex
# or, from a local clone:
#   powershell -ExecutionPolicy Bypass -File packaging\install-wavefield.ps1
#
# Safe to re-run: an existing clone is updated in place rather than
# re-cloned, and WhisperX setup already skips a working install.

param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "Wavefield-src"),
    [switch]$SkipWhisperX
)

$ErrorActionPreference = "Stop"
$repoUrl = "https://github.com/pcmflores19-ph/wavefield.git"

function Say($text)  { Write-Host $text }
function Step($text) { Write-Host ""; Write-Host "== $text" -ForegroundColor Cyan }

Say ""
Say "  Installing Wavefield from source"
Say "  ================================="
Say ""
Say "  This is the terminal alternative to the .exe installer - use it if"
Say "  Windows flagged the installer as a false positive. Everything it"
Say "  fetches comes from GitHub (source) and PyPI (packages); nothing here"
Say "  is a prebuilt executable you have to trust blindly."
Say ""

# ------------------------------------------------------------------- find git
Step "Checking for git"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Say "  git was not found - installing it."
        winget install --id Git.Git --scope user `
            --accept-package-agreements --accept-source-agreements
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" +
                    [Environment]::GetEnvironmentVariable("Path", "Machine")
    }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Say ""
        Say "  git is required. Install it from https://git-scm.com/download/win" -ForegroundColor Yellow
        Say "  then run this script again."
        Read-Host "  Press Enter to close"
        exit 1
    }
}
Say "  Found git"

# --------------------------------------------------------------- find Python
Step "Looking for Python"

# Same 3.9-3.12 range setup_whisperx.ps1 requires (WhisperX has no 3.13
# wheels yet) - matching it here means one Python search across the whole
# install instead of two different rules a reader has to reconcile.
$script:seenVersions = @()

function Find-Python {
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.12", "3.11", "3.10", "3.9")) {
            $candidates += ,@("py", @("-$v"))
        }
    }
    # The bare "python" stub Windows ships even with nothing installed opens
    # the Microsoft Store instead of running - Get-Command alone can't tell
    # the difference, so a real install is only trusted if it isn't sourced
    # from WindowsApps.
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pyCmd -and $pyCmd.Source -notlike "*\WindowsApps\*") {
        $candidates += ,@("python", @())
    }
    foreach ($c in $candidates) {
        try {
            $out = & $c[0] @($c[1] + @("-c", "import sys; print('%d.%d' % sys.version_info[:2])")) 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                $parts = $out.Trim().Split(".")
                $major = [int]$parts[0]; $minor = [int]$parts[1]
                $script:seenVersions += "$major.$minor"
                if ($major -eq 3 -and $minor -ge 9 -and $minor -le 12) {
                    return @{ Exe = $c[0]; Args = $c[1]; Version = $out.Trim() }
                }
            }
        } catch { }
    }
    return $null
}

$python = Find-Python
if (-not $python) {
    if ($script:seenVersions) {
        Say ("  Found Python " + (($script:seenVersions | Select-Object -Unique) -join ", ") +
             ", but Wavefield's transcription needs 3.9-3.12 specifically.")
    }
    Say ""
    Say "  Python 3.9-3.12 was not found on this computer." -ForegroundColor Yellow
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        $answer = Read-Host "  Install Python 3.12 now, automatically? [Y/n]"
        if ($answer -eq "" -or $answer -match "^[Yy]") {
            Step "Installing Python 3.12"
            winget install --id Python.Python.3.12 --scope user `
                --accept-package-agreements --accept-source-agreements
            $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" +
                        [Environment]::GetEnvironmentVariable("Path", "Machine")
            $python = Find-Python
        }
    }
    if (-not $python) {
        Say ""
        Say "  Install Python 3.9-3.12 from https://www.python.org/downloads/" -ForegroundColor Yellow
        Say "  Tick 'Add python.exe to PATH' during setup, then run this script again."
        Read-Host "  Press Enter to close"
        exit 1
    }
}
Say "  Found Python $($python.Version)"

# --------------------------------------------------------------- ffmpeg
Step "Checking for ffmpeg"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Say "  ffmpeg was not found - installing it."
        winget install --id Gyan.FFmpeg --scope user `
            --accept-package-agreements --accept-source-agreements
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" +
                    [Environment]::GetEnvironmentVariable("Path", "Machine")
    }
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        Say ""
        Say "  ffmpeg is required and was not found or installed automatically." -ForegroundColor Yellow
        Say "  Install it (e.g. 'winget install Gyan.FFmpeg') and make sure"
        Say "  ffmpeg.exe is on PATH, then run this script again."
        Read-Host "  Press Enter to close"
        exit 1
    }
}
Say "  Found ffmpeg"

# --------------------------------------------------------------- get the source
Step "Getting Wavefield's source"
if (Test-Path (Join-Path $InstallDir ".git")) {
    Say "  $InstallDir already exists - updating it"
    git -C $InstallDir pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw "git pull failed" }
} else {
    git clone $repoUrl $InstallDir
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}

# ------------------------------------------------------------- the environment
Step "Setting up Wavefield's dependencies"
$venvDir = Join-Path $InstallDir ".venv"
if (-not (Test-Path $venvDir)) {
    & $python.Exe @($python.Args + @("-m", "venv", $venvDir))
    if ($LASTEXITCODE -ne 0) { throw "could not create the environment" }
}
$venvPy = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "the environment is missing $venvPy" }

& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -r (Join-Path $InstallDir "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# ------------------------------------------------------------------ launcher
Step "Creating a launcher"
$launcher = Join-Path $InstallDir "run-wavefield.bat"
$venvPyw = Join-Path $venvDir "Scripts\pythonw.exe"
@"
@echo off
REM Generated by install-wavefield.ps1 - runs Wavefield from this source
REM checkout, inside its own virtual environment.
setlocal
cd /d "%~dp0auto_cut"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
"$venvPyw" app.py
endlocal
"@ | Set-Content -Path $launcher -Encoding ASCII
Say "  $launcher"

# ------------------------------------------------------------------ shortcut
Step "Adding a Start Menu shortcut"
try {
    $startMenu = [Environment]::GetFolderPath("StartMenu")
    $shortcutPath = Join-Path $startMenu "Wavefield.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $launcher
    $shortcut.WorkingDirectory = $InstallDir
    $icon = Join-Path $InstallDir "packaging\autocut.ico"
    if (Test-Path $icon) { $shortcut.IconLocation = $icon }
    $shortcut.Save()
    Say "  $shortcutPath"
} catch {
    Say "  Could not create a Start Menu shortcut - run $launcher directly instead." -ForegroundColor Yellow
}

# ------------------------------------------------------------------- WhisperX
if (-not $SkipWhisperX) {
    Step "Setting up speech recognition (WhisperX)"
    Say "  Same script the .exe installer runs - downloads about 2-3 GB the"
    Say "  first time, and does nothing if it's already set up."
    powershell -NoProfile -ExecutionPolicy Bypass `
        -File (Join-Path $InstallDir "packaging\setup_whisperx.ps1")
} else {
    Say ""
    Say "  Skipped WhisperX setup (-SkipWhisperX). Transcripts and subtitles"
    Say "  won't work until it's run - see Wavefield's File > Settings later."
}

Say ""
Say "  Done. Launch Wavefield from the Start Menu, or run:" -ForegroundColor Green
Say "    $launcher"
Say ""
