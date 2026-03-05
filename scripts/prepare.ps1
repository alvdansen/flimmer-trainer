# ======================================================================
# Flimmer Prepare Script (Windows)
# ======================================================================
#
# Runs latent and text pre-encoding for a training config or project.
# This is step 2 of the three-script workflow: setup.ps1 -> prepare.ps1 -> train.ps1
#
# Usage:
#   .\scripts\prepare.ps1 -Config path\to\train.yaml
#   .\scripts\prepare.ps1 -Project path\to\project.yaml
#   .\scripts\prepare.ps1 -Config path\to\train.yaml -DryRun
#   .\scripts\prepare.ps1 -Config path\to\train.yaml -Force
#
# Exactly one of -Config or -Project is required.
#
# ======================================================================

param(
    [string]$Config,
    [string]$Project,
    [switch]$DryRun,
    [switch]$Force,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Split-Path -Parent $ScriptDir

if ($Help) {
    Write-Host @"
Usage: .\scripts\prepare.ps1 [OPTIONS]

Run latent and text pre-encoding for a training config or project.

Options:
  -Config PATH    Direct path to a flimmer_train.yaml
  -Project PATH   Path to a project YAML (extracts base_config)
  -DryRun         Preview encoding without running it
  -Force          Re-encode latents even if cached
  -Help           Show this help message

Exactly one of -Config or -Project is required.

Examples:
  .\scripts\prepare.ps1 -Config config_templates\training\t2v_wan22.yaml
  .\scripts\prepare.ps1 -Project config_templates\projects\i2v_moe_phases.yaml
  .\scripts\prepare.ps1 -Config train.yaml -DryRun
  .\scripts\prepare.ps1 -Config train.yaml -Force
"@
    exit 0
}

# Validate: exactly one of -Config or -Project
if ($Config -and $Project) {
    Write-Host "ERROR: Cannot specify both -Config and -Project."
    exit 1
}

if (-not $Config -and -not $Project) {
    Write-Host "ERROR: One of -Config or -Project is required."
    Write-Host "Run .\scripts\prepare.ps1 -Help for usage."
    exit 1
}

# ── Activate Venv ───────────────────────────────────────────────────

$VenvDir = if ($env:FLIMMER_VENV) { $env:FLIMMER_VENV } else { Join-Path $RepoDir ".venv" }
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"

if (-not (Test-Path $ActivateScript)) {
    Write-Host "ERROR: Virtual environment not found at: $VenvDir"
    Write-Host ""
    Write-Host "Run setup.ps1 first to create the environment:"
    Write-Host "  .\scripts\setup.ps1 -Variant <variant>"
    Write-Host ""
    Write-Host "Or set `$env:FLIMMER_VENV to point to an existing venv."
    exit 1
}

& $ActivateScript

# ── Resolve Config Path ─────────────────────────────────────────────

if ($Project) {
    if (-not (Test-Path $Project)) {
        Write-Host "ERROR: Project file not found: $Project"
        exit 1
    }

    $Config = python -c @"
import yaml
from pathlib import Path
project_path = Path(r'$Project')
data = yaml.safe_load(open(project_path))
base = data.get('base_config', '')
if not base:
    import sys
    print('ERROR: project YAML has no base_config field', file=sys.stderr)
    sys.exit(1)
resolved = (project_path.parent / base).resolve()
print(resolved)
"@
    if ($LASTEXITCODE -ne 0) { exit 1 }

    Write-Host "Resolved config from project: $Config"
}

if (-not (Test-Path $Config)) {
    Write-Host "ERROR: Config file not found: $Config"
    exit 1
}

# ── Build Flags ─────────────────────────────────────────────────────

$ExtraFlags = @()
if ($DryRun) { $ExtraFlags += "--dry-run" }
if ($Force) { $ExtraFlags += "--force" }

# ── Run Encoding ────────────────────────────────────────────────────

Write-Host ""
Write-Host "=== Flimmer: Latent Pre-Encoding ==="
Write-Host ""
Write-Host "Config: $Config"
Write-Host ""

python -m flimmer.encoding cache-latents --config $Config @ExtraFlags
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "=== Flimmer: Text Pre-Encoding ==="
Write-Host ""

$TextFlags = @()
if ($DryRun) { $TextFlags += "--dry-run" }

python -m flimmer.encoding cache-text --config $Config @TextFlags
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "=== Encoding Complete ==="
Write-Host ""
Write-Host "Next step: Run .\scripts\train.ps1 to start training"
Write-Host ""
