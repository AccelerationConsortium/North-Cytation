<h1> Lash-E: A Friendly, Flexible Robotic Platform for Rapid Workflow Development  </h1>

All code written by Owen Melville with contributions by Ilya Yakavets, Monique Ngan, ChatGPT and Claude. Workflow code is written by individual users. 

<h2>The module (hardware)</h2>

Lash-E is a North Robotics liquid-handling robot plus the instruments around it. `Lash_E` in `master_usdl_coordinator.py` starts and coordinates all of them.

| Hardware | Controlled by | What it does |
|---|---|---|
| North robot arm (C9 controller) | `North_Robot` in `North_Safe.py` | Moves vials between racks, clamp and instruments; caps and decaps; pipettes with small and large tips; dispenses from reservoirs; vortexes |
| Clamp scale | `North_Robot` | Weighs the vial in the clamp (~0.2 mg); used to check dispensed volumes |
| North Track | `North_Track` in `North_Safe.py` | Moves wellplates between the pipetting area and the Cytation |
| Cytation 5 plate reader | `biotek_new.py` | Absorbance and fluorescence reads of wellplates |
| Heater and stirrer | `North_Temp` in `North_Safe.py` | Heats and stirs vials |
| Powder dispenser | `North_Powder` in `North_Safe.py` | Gravimetric powder dispensing |
| Spin coater | `North_Spin` in `North_Safe.py` (`lash.spinner`) | Spins a substrate held on a vacuum chuck; pneumatic lid |
| Photoreactor | `photoreactor_controller.py` | Light and stirring, via a Raspberry Pi Pico |

More detail: [system capabilities](docs/platform/system_capabilities.md). The reusable automation code is sorted into primitives, unit operations and flows in the [automation catalog](docs/platform/automation_catalog.md).

<h2>Running without hardware (simulation)</h2>

Every workflow can run without the robot. Set `SIMULATE = True` in the workflow (or in its `workflow_configs/<name>.yaml`) and run it from the repository root, e.g. `python workflows/workflow_template.py`. In simulation the robot controller is replaced by a mock, every automation step still runs, instruments may return no data, and no Slack messages are sent. The automated tests (`python -m pytest`) also run without hardware.

<h2>Run tracking</h2>

Every run gets a `run_id` and one row in `experiment_tracking/experiment_runs.csv` (workflow, start and stop times, status, log file, config file, output folder). The confirmed config of each run is saved as `logs/experiment_run_configs/<run_id>.yaml`. Workflows can call `lash.save_run_info(output_folder)` to put `config.yaml` and `run_info.yaml` in their own output folder.

<h2>File List:</h2>

- <b>North_Safe.py</b> Contains the reusable code for the control of the North Robot and the North Track which goes between the Cytation 5 and the North Robot. This is the most in-depth file.
  
- <b> biotek_new.py </b> Contains the code for control of the Cytation 5. 
  
- <b> master_usdl_coordinator.py </b> Contains reusable workflow code that uses multiple instruments: Cytation 5, North Robot & Track & Heater & Solid Dispenser, Photoreactor
   
- <b> photoreactor_controller </b> Controls the photoreactor via Raspberry Pi Pico. reactor_test.py is the local Pico program.
  
- <b> requirements.txt </b> What packages need installation to use all this? Minimum versions, for setting up a new machine.

- <b> environment.yml, requirements-lock.txt </b> The exact setup of the lab PC that runs the robot (conda environment `north_async_test`, Python 3.10.16), recorded 2026-11-16. To rebuild it: `conda env create -f environment.yml`. Two packages are not on the public package index and must be installed separately: the North Robotics SDK (`north==0.3.0.6`, from North Robotics) and `biotek_driver` (from `github.com/AC-SDL6/biotek_driver`). Re-record both files after installing or upgrading packages on the lab PC.
  
- <b> slack_agent.py </b> Controls messages sent to slack from the robot. Mostly written by ChatGPT.
  
- <b> vial_manager_gui.py </b> Helps the user ensure the initial state of the system matches the digital state

- <b> vial_csv.py </b> Reads and validates vial status CSVs, and saves them safely for the GUI.

- <b> workflow_config_manager.py </b> Creates, loads and saves each workflow's settings in `workflow_configs/<name>.yaml`.

- <b> CHANGELOG.md </b> Dated record of changes to the platform code.

<h2> Folder List: </h2>

The [repository organization assessment and proposed plan](docs/repository_organization.md)
records folder purposes and the staged, path-preserving cleanup proposal.
Approved moves and their caller/path updates are recorded there.

Documentation is grouped by purpose:
- [Workflow guides](docs/workflows/): running specific experiments and diagnostics.
- [Platform guides](docs/platform/): controllers, calibration, scheduler and labware.
- [Plans](docs/plans/): design proposals, with completed plans in its archive.
- [Archived documentation](docs/archive/): historical workflow summaries.
- [Papers](docs/papers/): reference literature.

The agent process guide and literature records live in the separate desktop
lab Slack agent checkout, not in this repository.

- <b> robot_state </b> Contains configuration files that abstract objects in the setup, such as pipet tips, racks, positions etc.

- <b> analysis </b>: Contains programs that analyze data from the robot to produce a result usable by the recommenders
 
- <b> recommenders: </b> Contains programs (eg using Baybe) to recommend conditions
  
- <b> status: </b> Because we move between different workflows, we keep digital versions of each workflows vial status state. 

- <b> inputs: </b> Input files for workflows, such as recipes and parameter lists.

- <b> workflow_configs: </b> Saved settings for each workflow (gitignored, local to the lab PC).

- <b> experiment_tracking: </b> Run tracking: `experiment_runs.csv` and the logger that writes it.

- <b> software_tests: </b> Automated regression tests using mocks, temporary files,
  or simulation instead of operating equipment. Run `python -m pytest` from the
  repository root; default discovery is restricted to this folder.

- <b> hardware_tests: </b> Operator-run equipment checks, even when they offer
  simulation. Run an individual program deliberately from the repository root.
  Some legacy checks operate hardware immediately when imported; never collect
  this folder with pytest or unittest.

- <b> tools: </b> Reusable operator utilities, not experiments or regression
  tests. Includes camera capture, utilization reporting, recovery/homing utilities,
  and the arm and track control GUIs. Run `python tools/robot_arm_control_gui.py`
  or `python tools/track_control_gui.py` using the robot's Python environment.
  `tools/book_robot.py` sends an actual booking email using the existing root
  secrets file; review its title/dates before explicitly running it.
  
- <b> workflows: </b> Eight active experimental/data-collection entrypoints and
  the workflow template. Shared support code lives in `workflows/helpers`;
  historical programs remain in `workflows/archive`. Helpers and the template
  are not scheduler jobs.

- <b> scheduler: </b> Workflow scheduling GUI and support. Launch with
  `python -m scheduler.scheduler_gui` from the repository root or the desktop lab
  launcher. The moved clamp, Cytation labware and vortex diagnostics remain
  schedulable from `hardware_tests` under their original names and saved configs.
  Other hardware programs are not automatically discovered. The vial-manager
  GUI and controllers remain at their existing paths.

- <b> source/firmware/photoreactor </b>: Source files uploaded to the photoreactor
  Pico: `reactor_test.py` and `photo_reactor.py`. Device-side filenames/imports
  are unchanged; the host controller uses the installed Pico modules, not this
  local folder. Its `lib.EMC2101` dependency must also be available on the device.
  Automation host controllers remain at root pending a separate migration.

- Powder-dispensing presets live in `robot_state/powder_settings.py`.

- <b> logs </b>: Contains logs from every digital and simulated run
  and retained LLM prompt/response history under `logs/LLM_prompts`.

- <b> output </b>: Contains output data. This is gitignored so that data remains locally. 
  Historical results are under `output/other`; scratch artifacts and coordinate
  exports are under `output/temp`. Previously tracked temp artifacts remain tracked
  pending their retention review.

- <b> pipetting_data/archives/recovery </b>: Local, ignored pipetting-recovery
  provenance and investigation snapshots.

<h2> Sample Workflow: </h2>

The sample workflow contained in this video is labelled and contains the programming elements in `workflows/archive/old_workflows/sample_workflow.py`. That file is out of date; start new workflows from `workflows/workflow_template.py`. 

[![Sample Workflow](https://img.youtube.com/vi/b5xtcwcI-eg/0.jpg)](https://www.youtube.com/watch?v=b5xtcwcI-eg)
