# Grid Generation Review

The relevant methods are:

## Grid generation

-   **`set_grid_origin()`** --- reads the current joint positions and
    uses `n9k.fk()` to determine the Cartesian X/Y origin.
-   **`_compute_grid_entries()`** --- calculates each target X/Y grid
    location, calls `n9k.ik()` to convert it into elbow and shoulder
    positions, selects between the two IK configurations, converts the
    angles to encoder counts, and verifies the result using `n9k.fk()`.
-   **`generate_grid_array()`** --- reads the grid dimensions and
    spacing, calls `_compute_grid_entries()`, and formats the resulting
    positions for use in `Locator.py`.

## Direct X/Y movement

-   `move_x_left()`
-   `move_x_right()`
-   `move_y_back()`
-   `move_y_forward()`

These four methods all call:

-   **`_move_xy_delta(dx, dy)`** --- calculates the current X/Y position
    using `self.robot.n9_fk()`, adds the requested X or Y displacement,
    then uses `self.robot.n9_ik()` to calculate the new elbow and
    shoulder positions.

The grid generation and direct X/Y movement therefore use essentially
the same **FK → target X/Y → IK** process and appear to share the same
underlying issue.
