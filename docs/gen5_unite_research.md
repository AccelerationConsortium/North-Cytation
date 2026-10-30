# UniteLabs Gen5 investigation

Reviewed 2026-10-30. The UniteLabs source review was research only. Subsequent
XML-only Gen5 tests are recorded below; none started a measurement or called
carrier/robot movement. Production reader and coordinator code was not changed
in this investigation.

## What UniteLabs establishes

The official example targets Gen5 3.14. It exposes a connector-level `protocol_controller.set_plate_type(plate_type, use_lid)` operation and demonstrates changing from 96-well to 24-well labware. The documentation says plate editing also works with protocols authored in Gen5. This establishes a third-party wrapper capability, not a native COM method with that name. [Official Gen5 connector documentation](https://docs.unitelabs.io/devices/biotek-gen5/#change-plate-type)

The example absorbance step contains `<Wells>Full Plate</Wells>`. Its documented `add_absorbance_step` arguments cover wavelengths, speed, index, and wavelength switching, with no wells argument. The published command list does not document `SetPartialPlate` or a comparable well-selection command. Therefore this example does not demonstrate partial reads after changing format. [Protocol controller examples and command reference](https://docs.unitelabs.io/devices/biotek-gen5/#protocol-controller)

Connector internals are not exposed by these SDK calls; connectors run separately and communicate through an API. Replacing the current driver with this SDK is not necessary to investigate the XML route. [Connector architecture](https://docs.unitelabs.io/concepts/integrate/connector/)

## Source availability checked

The public [UniteLabs GitLab projects endpoint](https://gitlab.com/api/v4/groups/unitelabs/projects?include_subgroups=true&per_page=100) returned 19 projects, including SDK/CDK repositories and some connectors, but no Gen5 connector. Public web searches did not locate its implementation. This is a statement about this search, not proof the implementation is unavailable through other channels. We cannot say how `set_plate_type()` is implemented internally or that it preserves runtime selection.

## Implication for the next test

UniteLabs does not supply the missing partial-well solution. The local programmer guide's `SetProcedure` documentation (pages 54–55) permits absorbance procedures and rejects unsupported tokens; Appendix A (pages 123–124) illustrates only `Full Plate`, not selected-well syntax. [Local programmer guide](</C:/Users/Imaging Controller/Desktop/Gen5 Automation Programmer Guide.pdf>)

Obtain a native procedure XML snapshot from an existing Gen5 protocol whose read step explicitly selects known wells. Use that returned encoding as evidence for a narrowly scoped XML-only test: change labware and embed A1 inside the read step, inspect readback, and validate. Do not claim this restores Runtime mode: it instead tests whether PerStep mode can still execute the requested subset. Do not start a read, load a carrier, or replace the normal driver during this investigation.

## PyLabRobot implementation

The current BioTek driver accepts a plate object and a list of wells. It sends
plate dimensions through the firmware `y` command. For absorbance it decomposes
the requested wells into rectangular regions and encodes each region's row and
column bounds into a direct `D` instrument command. It does not call Gen5's
SetProcedure or SetPartialPlate. This is a working implementation strategy for
explicit well subsets, but does not reveal a way to restore runtime selection
in the COM API. Adopting it would change the reader driver and requires separate
verification for this instrument, transport and assays. It is not a drop-in XML
fix. [PyLabRobot source: set_plate and read_absorbance](https://github.com/PyLabRobot/pylabrobot/blob/main/pylabrobot/agilent/biotek/plate_reader_base.py#L399)

## XML-only tests performed after the research

The existing local `C:\Protocols\Serena_Test_A1A3.prt` returned native XML with
`WellSelection=PerStep` and `<ReadStep><Wells>A1..A3</Wells>`. This was obtained
using GetProcedure, with no read or motion. Two further protocols could not
be opened in that same session because Gen5 reported a document already in
memory; the diagnostic did not add a close or reset to force them open.
[Saved native XML](../output/cytation_xml_only/20261030T170603_463115Z_known_wells/Serena_Test_A1A3.xml)

| Attempt | Gen5 result | Saved evidence |
| --- | --- | --- |
| Existing turbidity XML, 48-well labware, unsupported top-level tokens omitted, `Wells=A1..A3` | SetProcedure rejected the Wells value as unexpected | [Range test](../output/cytation_xml_only/20261030T170653_416677Z/summary.json) |
| Same, `Wells=A1` | SetProcedure rejected the Wells value as unexpected | [Single-well test](../output/cytation_xml_only/20261030T170705_272938Z/summary.json) |
| NewExperimentEx with protocol-metadata XML containing 48-well labware and original Runtime selection | Rejected WellSelection as unsupported | [Metadata test](../output/cytation_xml_only/20261030T170732_553325Z_new_experiment_ex/summary.json) |

Each folder retains the exact setter input and driver errors. These tests rule
out those specific encodings/routes on the installed Gen5. They do not prove
that every possible partial-well input format is unsupported. In particular,
getter output is not a guarantee of setter input support.

## Recommendation and precise unanswered question

Do not enable XML plate replacement in the production workflow yet. The
requirement is requested-well measurement, not a full-plate read with unwanted
results discarded. The investigated public UniteLabs documentation and tested
COM routes do not supply that combination. Retain the existing working path.

Send the following question with the saved XML to Agilent/Gen5 support or ask
UniteLabs for the actual implementation (no message has been sent):

> On Gen5 3.15, how can automation change a loaded absorbance protocol from
> 96 to 48 wells while measuring only requested wells? SetProcedure accepts the
> labware change but defaults WellSelection to PerStep, causing SetPartialPlate
> to fail with -2581. Both SetProcedure and NewExperimentEx reject WellSelection.
> SetProcedure also rejects Wells values A1 and A1..A3, including a range emitted
> by GetProcedure from a native partial-read protocol. What supported input
> schema or API preserves runtime selection or sets a partial region per step?

The diagnostic script now exposes `--step-wells A1` and `--step-wells A1..A3` only
as explicit experimental modes; both are known to be rejected on this machine.
Five diagnostic tests pass with mocked Gen5, which verifies artifact handling
and call paths, not native API support.
