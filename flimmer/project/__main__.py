"""CLI for project-based training management.

Bridges project YAML files to the existing training CLI. The project
YAML defines multi-phase training (unified -> expert fork) while the
training CLI accepts single-config YAML files. This CLI generates
per-phase configs and drives execution.

Commands:
    run      Run next pending phase (or all with --all)
    status   Show project phase status
    plan     Dry-run: show what phases would run and their overrides

Usage::

    python -m flimmer.project run --project path/to/project.yaml
    python -m flimmer.project status --project path/to/project.yaml
    python -m flimmer.project plan --project path/to/project.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> None:
    """Run next pending phase (or all with --all).

    For each pending phase:
    1. Generate merged config via merge_phase_config()
    2. Mark phase RUNNING, save project
    3. Execute training via subprocess
    4. Mark COMPLETED on success, exit on failure
    """
    from flimmer.phases import PhaseStatus

    from flimmer.project.loader import (
        load_project_yaml,
        merge_phase_config,
        project_from_yaml,
    )

    yaml_path = Path(args.project)
    project = project_from_yaml(yaml_path)
    project_dir = yaml_path.parent

    # Get base_config path from the project YAML
    project_data = load_project_yaml(yaml_path)
    base_config = project_data.get("base_config")
    if base_config is None:
        print("Error: project YAML must specify 'base_config'.", file=sys.stderr)
        sys.exit(1)

    base_config_path = (project_dir / base_config).resolve()
    if not base_config_path.exists():
        print(
            f"Error: base_config not found: {base_config_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path("./output") / project.name
    output_dir.mkdir(parents=True, exist_ok=True)

    ran_any = False
    for i, entry in enumerate(project.phases):
        if entry.status == PhaseStatus.COMPLETED:
            print(f"  Phase {i}: {entry.config.display_name} -- COMPLETED (skipped)")
            continue
        if entry.status == PhaseStatus.RUNNING:
            print(f"  Phase {i}: {entry.config.display_name} -- RUNNING (resuming)")
        if entry.status in (PhaseStatus.PENDING, PhaseStatus.RUNNING):
            # Generate merged config for this phase
            merged_config_path = output_dir / f".flimmer_phase_{i}_config.yaml"
            merge_phase_config(base_config_path, project, i, merged_config_path)

            # Mark phase running and save
            if entry.status == PhaseStatus.PENDING:
                project.mark_phase_running(i)
                project.save(project_dir)

            print(f"\n  Running Phase {i}: {entry.config.display_name}")
            print(f"  Config: {merged_config_path}")

            # Invoke the training CLI
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "flimmer.training",
                    "train",
                    "--config",
                    str(merged_config_path),
                ],
                check=False,
            )

            if result.returncode == 0:
                project.mark_phase_completed(i)
                project.save(project_dir)
                print(f"  Phase {i}: {entry.config.display_name} -- COMPLETED")
            else:
                print(
                    f"  Phase {i}: {entry.config.display_name} -- FAILED (exit code {result.returncode})",
                    file=sys.stderr,
                )
                sys.exit(1)

            ran_any = True
            if not args.all:
                break

    if not ran_any:
        print("  All phases completed. Nothing to run.")


def cmd_status(args: argparse.Namespace) -> None:
    """Show project phase status.

    Loads the project and prints each phase with its index,
    status, and display name.
    """
    from flimmer.project.loader import project_from_yaml

    yaml_path = Path(args.project)
    project = project_from_yaml(yaml_path)

    print(f"\nProject: {project.name}")
    print(f"Model:   {project.model_id}")
    print(f"Phases:  {len(project.phases)}")
    print()

    for i, entry in enumerate(project.phases):
        status = entry.status.value.upper()
        name = entry.config.display_name or entry.config.phase_type
        print(f"  [{status:9s}] Phase {i}: {name}")


def cmd_plan(args: argparse.Namespace) -> None:
    """Dry-run: show the fully resolved training plan with project overrides.

    For each pending phase, merges the base config with phase overrides
    and resolves training parameters. Shows the same detailed plan output
    as `python -m flimmer.training plan` but with project overrides applied.

    Falls back to showing raw overrides if the base config is not available.
    """
    from flimmer.phases import PhaseStatus

    from flimmer.project.loader import (
        load_project_yaml,
        project_from_yaml,
    )

    yaml_path = Path(args.project)
    project = project_from_yaml(yaml_path)
    project_dir = yaml_path.parent

    # Try to resolve the full plan via the training config pipeline
    project_data = load_project_yaml(yaml_path)
    base_config = project_data.get("base_config")
    base_config_path = None
    if base_config:
        candidate = (project_dir / base_config).resolve()
        if candidate.exists():
            base_config_path = candidate

    if base_config_path is not None:
        _plan_resolved(project, project_data, base_config_path)
    else:
        _plan_simple(project)


def _plan_resolved(
    project: object, project_data: dict, base_config_path: Path
) -> None:
    """Show the fully resolved training plan by merging base config with overrides."""
    import tempfile

    from flimmer.phases import PhaseStatus
    from flimmer.project.loader import merge_phase_config

    # Show completed/running phases briefly
    for i, entry in enumerate(project.phases):
        if entry.status != PhaseStatus.PENDING:
            status = entry.status.value.upper()
            name = entry.config.display_name or entry.config.phase_type
            print(f"  Phase {i}: {name} [{status}] -- skip")

    # Resolve each pending phase through the training config pipeline
    all_resolved_phases = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, entry in enumerate(project.phases):
            if entry.status != PhaseStatus.PENDING:
                continue

            merged_path = Path(tmpdir) / f"phase_{i}.yaml"
            merge_phase_config(base_config_path, project, i, merged_path)

            try:
                from flimmer.config.training_loader import load_training_config
                from flimmer.training.phase import resolve_phases

                config = load_training_config(str(merged_path))
                phases = resolve_phases(config)
                all_resolved_phases.extend(phases)
            except Exception as e:
                name = entry.config.display_name or entry.config.phase_type
                print(f"\n  Phase {i}: {name} -- ERROR resolving: {e}")

    if all_resolved_phases:
        from flimmer.training.logger import TrainingLogger
        logger = TrainingLogger(backends=["console"])
        logger.print_training_plan(all_resolved_phases)


def _plan_simple(project: object) -> None:
    """Fallback plan display when base config is not available."""
    from flimmer.phases import PhaseStatus

    print(f"\nProject: {project.name}")
    print(f"Model:   {project.model_id}")
    print()

    pending_count = sum(
        1 for e in project.phases if e.status == PhaseStatus.PENDING
    )
    print(f"Pending phases: {pending_count} of {len(project.phases)}")
    print()

    for i, entry in enumerate(project.phases):
        status = entry.status.value.upper()
        name = entry.config.display_name or entry.config.phase_type

        if entry.status != PhaseStatus.PENDING:
            print(f"  Phase {i}: {name} [{status}] -- skip")
            continue

        print(f"  Phase {i}: {name} [{status}] -- would run")
        if entry.config.overrides:
            for key, value in entry.config.overrides.items():
                print(f"    {key}: {value}")
        if entry.config.extras:
            for key, value in entry.config.extras.items():
                print(f"    (extra) {key}: {value}")
        print()


# ── Parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the project CLI.

    Subcommands:
        run     -- Run next pending phase (or all with --all)
        status  -- Show phase statuses
        plan    -- Dry-run showing what would happen
    """
    parser = argparse.ArgumentParser(
        prog="python -m flimmer.project",
        description="Flimmer project management -- multi-phase training from YAML.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ─── run ───
    run_parser = subparsers.add_parser(
        "run",
        help="Run next pending phase (or all with --all).",
    )
    run_parser.add_argument(
        "--project", "-p",
        required=True,
        help="Path to the project YAML file.",
    )
    run_parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run all pending phases (default: run next one only).",
    )
    run_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for merged configs and training output. "
        "Default: ./output/{project.name}",
    )

    # ─── status ───
    status_parser = subparsers.add_parser(
        "status",
        help="Show project phase status.",
    )
    status_parser.add_argument(
        "--project", "-p",
        required=True,
        help="Path to the project YAML file.",
    )

    # ─── plan ───
    plan_parser = subparsers.add_parser(
        "plan",
        help="Dry-run: show what phases would run.",
    )
    plan_parser.add_argument(
        "--project", "-p",
        required=True,
        help="Path to the project YAML file.",
    )

    return parser


def main() -> None:
    """Entry point for the project CLI."""
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "run": cmd_run,
        "status": cmd_status,
        "plan": cmd_plan,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
