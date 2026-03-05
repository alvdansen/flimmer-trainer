# ======================================================================
# Flimmer Training Script (Windows)
# ======================================================================
#
# Launches training via project-based phase management or single config.
# This is step 3 of the three-script workflow: setup.ps1 -> prepare.ps1 -> train.ps1
#
# Usage:
#   .\scripts\train.ps1 -Project path\to\project.yaml
#   .\scripts\train.ps1 -Project path\to\project.yaml -All
#   .\scripts\train.ps1 -Project path\to\project.yaml -Status
#   .\scripts\train.ps1 -Config path\to\train.yaml
#   .\scripts\train.ps1 -Config path\to\train.yaml -DryRun
#
# Exactly one of -Config or -Project is required.
#
# ======================================================================

param(
    [string]$Config,
    [string]$Project,
    [switch]$All,
    [switch]$DryRun,
    [switch]$Status,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Split-Path -Parent $ScriptDir

if ($Help) {
    Write-Host @"
Usage: .\scripts\train.ps1 [OPTIONS]

Launch Flimmer training with project phase management or single config.

Options:
  -Project PATH   Path to a project YAML (multi-phase training)
  -Config PATH    Direct path to a flimmer_train.yaml (single run)
  -All            Run all pending phases (only with -Project)
  -DryRun         Show what would happen without training
  -Status         Show project phase statuses (only with -Project)
  -Help           Show this help message

Exactly one of -Config or -Project is required.

Examples:
  .\scripts\train.ps1 -Project config_templates\projects\i2v_moe_phases.yaml
  .\scripts\train.ps1 -Project config_templates\projects\i2v_moe_phases.yaml -All
  .\scripts\train.ps1 -Config config_templates\training\t2v_wan22.yaml
  .\scripts\train.ps1 -Project config_templates\projects\i2v_moe_phases.yaml -Status
  .\scripts\train.ps1 -Project config_templates\projects\i2v_moe_phases.yaml -DryRun
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
    Write-Host "Run .\scripts\train.ps1 -Help for usage."
    exit 1
}

if ($All -and -not $Project) {
    Write-Host "ERROR: -All can only be used with -Project."
    exit 1
}

if ($Status -and -not $Project) {
    Write-Host "ERROR: -Status can only be used with -Project."
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

# ── CUDA Memory Optimization ───────────────────────────────────────

if (-not $env:PYTORCH_CUDA_ALLOC_CONF) {
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
}

# ── Handle -Status ──────────────────────────────────────────────────

if ($Status) {
    python -m flimmer.project status --project $Project
    exit $LASTEXITCODE
}

# ── Handle -DryRun ──────────────────────────────────────────────────

if ($DryRun) {
    if ($Project) {
        python -m flimmer.project plan --project $Project
    } else {
        python -m flimmer.training plan --config $Config
    }
    exit $LASTEXITCODE
}

# ── Run Training ────────────────────────────────────────────────────

$ExitCode = 0

if ($Project) {
    Write-Host ""
    Write-Host "=== Flimmer: Project Training ==="
    Write-Host ""
    Write-Host "Project: $Project"
    Write-Host ""

    $Cmd = @("-m", "flimmer.project", "run", "--project", $Project)
    if ($All) { $Cmd += "--all" }

    python @Cmd
    $ExitCode = $LASTEXITCODE
} else {
    Write-Host ""
    Write-Host "=== Flimmer: Single Config Training ==="
    Write-Host ""
    Write-Host "Config: $Config"
    Write-Host ""

    python -m flimmer.training train --config $Config
    $ExitCode = $LASTEXITCODE
}

if ($ExitCode -eq 0) {
    Write-Host ""
    Write-Host "=== Training Complete ==="
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "=== Training Failed (exit code $ExitCode) ==="
    Write-Host ""
}

exit $ExitCode
