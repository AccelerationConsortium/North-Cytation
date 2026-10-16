"""Prepare the output migration; no data is moved by this script."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'
moves = []
for entry in sorted(OUT.iterdir()):
    name = entry.name
    parent = None
    if name.startswith('fluorescence_calibration_'):
        parent = 'fluorescence_calibration'
    elif name.startswith('calibration_session_'):
        parent = 'pipette_calibration/sessions'
    elif name.startswith('surfactant_grid_'):
        parent = 'surfactant_grid/legacy'
    elif name.startswith('validation_'):
        parent = 'surfactant_grid/validation'
    elif name.startswith('water_validation_'):
        parent = 'water_validation'
    elif name.startswith('pipetting_validation_'):
        parent = 'color_mixing/validation'
    elif name.startswith('color_mixing_results_'):
        parent = 'color_mixing/results'
    elif name.startswith('two_point_series_demo_'):
        parent = 'two_point_calibration/results'
    elif name.startswith('calibration_vials_short_mass_validation'):
        parent = 'calibration_vials_short_mass_validation'
    elif name.startswith('ilya_dispense_timing_'):
        parent = 'ilya_dispense_timing'
    exact = {
        'experimental_surfactant_grid': 'surfactant_grid/experimental',
        'simulated_surfactant_grid': 'surfactant_grid/simulated',
        'Interrupted_Runs': 'surfactant_grid/interrupted',
        'calibration_runs': 'pipette_calibration/legacy_runs',
        'calibration_v2_runs': 'pipette_calibration/legacy_v2_runs',
        'two_point_analysis': 'two_point_calibration/analysis',
        'two_point_focus_glycerol_25_50_75_100': 'two_point_calibration/two_point_focus_glycerol_25_50_75_100',
    }
    destination = OUT / exact[name] if name in exact else OUT / parent / name if parent else None
    if destination:
        moves.append({'source': str(entry.relative_to(ROOT)).replace('\\', '/'),
                      'destination': str(destination.relative_to(ROOT)).replace('\\', '/')})

def edit(relative, replacements):
    path = ROOT / relative
    original = path.read_bytes()
    text = original.decode('utf-8')
    for before, after in replacements:
        if before not in text:
            raise ValueError(f'Missing expected text in {relative}: {before!r}')
        text = text.replace(before, after)
    if text.encode('utf-8') != original:
        save_original(path)
        path.write_bytes(text.encode('utf-8'))
        changed.add(relative)

changed = set()
# Preserve the original user-edited sources before applying this migration.
backup = ROOT / 'temp/output_reorganization_source_backup'
def save_original(path):
    destination = backup / path.relative_to(ROOT)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())

edit('workflows/fluorescence_calibration_workflow.py', [
    ('Path("output") / ("fluorescence_calibration_', 'Path("output") / "fluorescence_calibration" / ("fluorescence_calibration_')])
for relative in ['workflows/surfactant_grid_adaptive_concentrations.py',
                 'workflows/surfactant_multidimensional_workflow.py',
                 'workflows/surfactant_grid_ailsa.py']:
    edit(relative, [('"simulated_surfactant_grid"', '"surfactant_grid/simulated"'),
                    ('"experimental_surfactant_grid"', '"surfactant_grid/experimental"')])
edit('workflows/surfactant_grid_adaptive_concentrations.py', [
    ('f"output/validation_', 'f"output/surfactant_grid/validation/validation_')])
edit('workflows/surfactant_multidimensional_workflow.py', [
    ('"output", f"validation_', '"output", "surfactant_grid", "validation", f"validation_')])

# Update references to moved data in source and documentation, never historical
# result files, run snapshots, logs, or raw measurements.
paths = set()
for directory in ['workflows', 'analysis', 'scripts', 'docs', 'tests']:
    for path in (ROOT / directory).rglob('*'):
        if path.suffix in {'.py', '.md', '.yaml', '.yml', '.json'} and not any(
                part in {'output', '__pycache__', '.pytest_cache'} for part in path.relative_to(ROOT).parts):
            paths.add(path)
for path in paths:
    data = path.read_bytes()
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        continue
    original = text
    for move in sorted(moves, key=lambda item: len(item['source']), reverse=True):
        for separator in ['/', '\\\\', '\\']:
            old = move['source'].replace('/', separator)
            new = move['destination'].replace('/', separator)
            text = text.replace(old, new)
    # Paths containing future run names need the same parent change.
    text = text.replace('output/simulated_surfactant_grid', 'output/surfactant_grid/simulated')
    text = text.replace('output/experimental_surfactant_grid', 'output/surfactant_grid/experimental')
    if text != original:
        save_original(path)
        path.write_bytes(text.encode('utf-8'))
        changed.add(str(path.relative_to(ROOT)).replace('\\', '/'))

# Archived producers remain usable if explicitly launched again.
edit('workflows/archive/color_mixing.py', [
    ('import sys\r\n', 'import sys\r\nimport os\r\n'),
    ('f"output/pipetting_validation_', 'f"output/color_mixing/validation/pipetting_validation_'),
    ('f"../utoronto_demo/output/color_mixing_results_', 'f"output/color_mixing/results/color_mixing_results_'),
    ('        results_filename = ', '        os.makedirs("output/color_mixing/results", exist_ok=True)\r\n        results_filename = ')])
edit('workflows/archive/calibration_vials_short_mass_validation.py', [
    ('Path(__file__).resolve().parent.parent / "output"',
     'Path(__file__).resolve().parents[2] / "output" / "calibration_vials_short_mass_validation"')])
edit('workflows/archive/ilya_workflow_v2.py', [
    ('os.makedirs("output", exist_ok=True)', 'os.makedirs("output/ilya_dispense_timing", exist_ok=True)'),
    ('os.path.join("output", f"ilya_dispense_timing_', 'os.path.join("output", "ilya_dispense_timing", f"ilya_dispense_timing_')])
# A separate config redirects the main workflow without changing the standalone
# calibration program, its config, or its hardcoded GUI discovery paths.
config = ROOT / 'sdl_pipette_calibration/experiment_config_workflow.yaml'
if config.exists():
    raise ValueError('Workflow-specific config already exists; review before replacing')
config.write_bytes((ROOT / 'sdl_pipette_calibration/experiment_config.yaml').read_bytes().replace(
    b'base_directory: output/', b'base_directory: ../output/pipette_calibration/runs/'))
changed.add('sdl_pipette_calibration/experiment_config_workflow.yaml')
edit('workflows/pipette_calibration_workflow.py', [
    ('CALIBRATION_CONFIG_FILE = "sdl_pipette_calibration/experiment_config.yaml"',
     'CALIBRATION_CONFIG_FILE = "sdl_pipette_calibration/experiment_config_workflow.yaml"')])

(ROOT / 'temp/output_move_plan.json').write_text(json.dumps(moves, indent=2), encoding='utf-8')
(ROOT / 'temp/output_changed_files.json').write_text(json.dumps(sorted(changed), indent=2), encoding='utf-8')
print(json.dumps({'planned_moves': len(moves), 'changed_files': sorted(changed)}, indent=2))
