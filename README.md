<h1> Lash-E: A Friendly, Flexible Robotic Platform for Rapid Workflow Development  </h1>

All code written by Owen Melville with contributions by Ilya Yakavets, Monique Ngan, ChatGPT and Claude. Workflow code is written by individual users. 


<h2>File List:</h2>

- <b>North_Safe.py</b> Contains the reusable code for the control of the North Robot and the North Track which goes between the Cytation 5 and the North Robot. This is the most in-depth file.
  
- <b> biotek_new.py </b> Contains the code for control of the Cytation 5. 
  
- <b> master_usdl_coordinator.py </b> Contains reusable workflow code that uses multiple instruments: Cytation 5, North Robot & Track & Heater & Solid Dispenser, Photoreactor
   
- <b> photoreactor_controller </b> Controls the photoreactor via Raspberry Pi Pico. reactor_test.py is the local Pico program.
  
- <b> requirements.txt </b> What packages need installation to use all this?
  
- <b> slack_agent.txt </b> Controls messages sent to slack from the robot. Mostly written by ChatGPT.
  
- <b> vial_manager_gui.py </b> Helps the user ensure the initial state of the system matches the digital state

<h2> Folder List: </h2>

The [repository organization assessment and proposed plan](docs/repository_organization.md)
records folder purposes and the staged, path-preserving cleanup proposal.
Moves require approval; existing runtime paths are unchanged.

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
  
- <b> workflows: </b> Contains active workflows that are designed for the setup, such as the color matching workflow.

- <b> photo-reactor </b>: Contains the RPi program for the photoreactor.

- <b> logs </b>: Contains logs from every digital and simulated run
  and retained LLM prompt/response history under `logs/LLM_prompts`.

- <b> output </b>: Contains output data. This is gitignored so that data remains locally. 
  Historical results are under `output/other`; scratch artifacts and coordinate
  exports are under `output/temp`. Previously tracked temp artifacts remain tracked
  pending their retention review.

- <b> pipetting_data/archives/recovery </b>: Local, ignored pipetting-recovery
  provenance and investigation snapshots.

<h2> Sample Workflow: </h2>

The sample workflow contained in this video is labelled and contains the programming elements in the sample_workflow.py program in the "workflows" directory. 

[![Sample Workflow](https://img.youtube.com/vi/b5xtcwcI-eg/0.jpg)](https://www.youtube.com/watch?v=b5xtcwcI-eg)
