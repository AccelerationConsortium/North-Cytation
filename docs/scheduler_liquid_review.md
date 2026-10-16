# Scheduler liquid dispensing review

Run **Simulate**, then click **Review liquids** in the workflow row's **Liquids** column.
The dialog reads that result's workflow log; it does not derive liquid usage from vial
names or workflow configuration. An incomplete result may show only partial usage.

Each liquid has Calibration Summary, Coverage, Past Calibration Results, Past Validation,
Overvolume Validation, Environment, and Recommendations tabs. Calibration Summary contains the
tip-specific coverage graph and measured-versus-target plot from the selected optimal
conditions file. Coverage contains the workflow volume table and log line references.
Past Calibration Results lets the user select a saved optimal-conditions result or
saved `raw_validation_data.csv` result. It plots measured against target volume,
lists the saved measurements, and shows the pipetting parameters at each target.
Calibration search trials from `raw_measurements.csv` are not plotted. Older runs
in `output/calibration_runs` use a different optimal-conditions schema and are
supported. Only confirmed non-simulated runs are shown: recorded execution summaries
take priority over requested config mode, and runs with unknown mode are omitted.
With no selected calibration, coverage, past-run browsing,
and recommendations remain.
Multiple candidate calibration files require an explicit selection. Backups are excluded.

## Log interpretation

The parser counts destination operations: vial dispensing, wellplate transfer and
pipetting messages, and reservoir operations. It does not count high-level transfer
summaries or pump movements, which include air gaps and overaspirate. The log gives
commanded volumes, not proof of successful physical delivery. Older logs round volumes
to 0.001 mL. Each operation retains its source line for inspection.

Liquid classes come from calibration messages or the new `Liquid review context` log
entry emitted by parameter resolution. The class follows the aspirated fluid across
multiple destination operations. Missing classes are explicitly unidentified.

## Coverage and matching

Coverage is separated at 200 µL: small tips below 200, large tips at or above 200.
Interpolation never bridges this boundary. Logged tips that disagree with the implied
calibration regime are flagged. Reservoir usage requires separate calibration support.
Coverage describes the available calibration points, not guaranteed performance.

Validation matching compares every PipettingParameters field except overaspirate,
using current calibration interpolation within the applicable regime and runtime
defaults for fields absent from the calibration CSV. Numeric tolerance is 1e-6 relative
and 1e-8 absolute in stored units. Missing measurement settings are excluded. Historical
tip identity is inferred because the master CSV has no tip column. Session summary rows
are not added as observations, avoiding double counting of individual measurements.
Exact duplicate measurement rows are also removed. Exclusion counts show the first
setting mismatch for each rejected record.

## Statistical display

### Past Validation

This tab reads embedded measurements from `master_pipetting_measurements.csv`, using
only explicitly accepted stages that match current settings under the rules above.
Overaspirate may vary, including at interpolated target volumes. Unknown acceptance
decisions and unselected stages are excluded and counted. Explicitly marked simulated
measurements are excluded; records without simulation metadata cannot be classified
by execution mode. Acceptance provenance is documented in
`embedded_validation_decisions.md`.

At each target volume, the plot shows accepted session means and their equally weighted
historical mean. The 95% Student-t confidence interval uses variation between session
means, so sessions with more replicates do not receive extra weight. One session has
no historical confidence interval; intervals based on few sessions are uncertain.
The interval estimates the mean, not the range of future individual dispenses.
Accepted out-of-tolerance measurements remain included.

The table shows mean measured volume, signed error, confidence interval, session and
measurement counts, and latest validation date. Select a target to inspect its accepted
measurements and decision metadata. The graph toolbar supports zooming and saving.

### Overvolume Validation

For each measured target volume, average observations by session and tested V_OA, fit
signed delivery error versus V_OA, and find the zero-error crossing. Only positive slopes
with a crossing inside the tested range yield a point estimate. Three or more sessions
are required for an exploratory 95% interval, computed by 300 deterministic session
bootstrap samples. Unstable bootstrap slopes suppress the interval. This local linear
model remains exploratory; it has not been validated as a production compensation rule.
Intervals describe estimated optimal V_OA, not future dispensing variability.

High-error observations have smaller visual markers but remain in the regression;
their errors contain information about response. One tested V_OA is not a warning and
cannot independently identify an optimum. No calibration files are automatically changed.

The black cross is the raw current calibration V_OA, before optional runtime overvolume
compensation. The red dot is the estimated zero-error setting. The Overvolume Validation
table contains only historical target volumes with matching measurement records; it
marks which are used exactly by this workflow. It shows the latest validation date,
signed mean error, and 95th percentile absolute error for each volume. Individual
measurements, including their signed and absolute errors, can be opened from that tab.
Recommendations are gathered in their own tab.

## Environment, original runs, and remaining work

Environmental records from parameter_corrections.csv are joined by session and target
volume. Historical appended temperature/humidity/pressure columns without matching
headers are supported. These are correction-time readings. Additional environmental
data can be obtained and cross-referenced, but ingestion of external sensor history,
timestamp alignment, and multivariable environmental modeling are not implemented yet.

Past optimal conditions are read from `sdl_pipette_calibration/output` and
`output/calibration_runs` / `output/calibration_v2_runs`; saved validation results
are read from `sdl_pipette_calibration/output` and `output/validation_runs`.
The validation file's `parameters_used` field and the calibration file's hardware
parameter columns supply the settings shown for each target volume.
The Overvolume Validation tab instead reads embedded validation measurements from
`pipetting_data/master_pipetting_measurements.csv`. Past calibration runs remain
separate from current-settings estimates: source-run
provenance is not preserved in the current imported calibration files.

Calibration hashes are captured when the scheduler starts a job. A changed file produces
a stale-simulation notice. Refresh reloads current data; rerunning simulation updates the
workflow evidence. Unrecognized log formats and unidentified classes require inspection;
the review is advisory and does not gate execution.

Tests: `python -m pytest tests/test_liquid_review.py tests/test_scheduler_simulation_gui.py tests/test_tip_selection.py tests/test_vial_transfer_chunk_calibration.py -q`
