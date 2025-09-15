# Status Files on the North-Cytation set-up
_Last updated: Jun 10, 2025_

Necessary files for each workflow:
1. A vial input `.csv` file (path is inputted by the user)
2. `robot_status.yaml` -- stores current state of the robot arm and the deck
3. `track_status.yaml` -- stores the state of wellplates that are moved by the gripper

Many of the issues we have encountered when running workflows were related to poor formatting of status files, so here is a guide for formatting your status files to prevent these errors.
Although it sounds obvious, remember to **save your files** after updating them before running workflows.

## How to format the files
### Vial Input CSV file
This file should be formatted with these column names:
|vial_index|vial_name|location|location_index|vial_volume|capped|cap_type|vial_type|home_location|home_location_index|
|----|----|----|----|----|----|----|----|----|----|
|0|source_vial_a|main_8mL_rack|0|3.5|True|open|8_mL|main_8mL_rack|0|

- vial_index: a _unique_ numerial index used for indenfying the vial
- vial_name: a _unique_ name for identifying the vial (vials can be accessed using vial_name or vial_index)
- location: the labware where the vial is currently located (ex. main_8mL_rack, large_vial_rack (20mL vial rack), small_vial_rack (2mL vial rack))
- location_index: where the vial is within the location / racks
  - 0 if there is only 1 position in that location (like in the clamp)
- vial_volume: the volume of liquid (in mL) currently in the vial
- capped: True/False (True if capped, False if not), this determines if it can be moved by the gripper
- cap_type:  `open` if using the septa cap with the septa removed (ensure that capped=True) or `False` if not capped
- vial_type: 8_mL, 20_mL
- home_location: the location where the vial should be returned to
- home_location_index: the index within the home_location to return to
  **the difference between location and home_location is that the location will be updated in real time while the vial is moved, while home_location does not. If the starting location of the vial is already considered the "home_location", ensure both are the same before running the workflow

You can use `North_Robot.check_input_file()` for a display of the inputted values in the terminal and in a GUI to confirm before running the workflow.

### track_status.yaml
Please ensure that this file is accurate before starting workflows as the wrong movements of the track can break wellplates.

The file contains the following inputs:
- nr_occupied: True if the wellplate stand for the north robot to pipet into is occupied, False if not
- num_in_source: the number of wellplates in the source wellplate stack
- num_in_waste: the number of wellplates in the waste wellplate stack
- wellplate_type: "48 WELL PLATE" or "96 WELL PLATE"

You can use `North_Track.check_input_file()` for a display of the inputted values in the terminal and in a GUI to confirm before running the workflow.

### robot_status.yaml 
This file typically does not need to be edited except for when refilling pipet tips (run `refill_tips.py`)

