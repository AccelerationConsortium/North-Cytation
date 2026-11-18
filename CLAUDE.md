# Lash-E (North + Cytation module)

Read `.github/copilot-instructions.md` first: it is the single set of instructions for AI
assistants working in this repository (GitHub Copilot reads it directly).

Writing a new workflow: start from `workflows/workflow_template.py` and build it from the flows
in `docs/platform/automation_catalog.md`. Every workflow passes `liquid=` to each dispense, uses
stock vials with a `stock_id`, and calls `lash_e.save_run_info(output_dir)` once its output
folder exists (sample tracking). Simulate first; never skip automation steps when `SIMULATE=True`.
