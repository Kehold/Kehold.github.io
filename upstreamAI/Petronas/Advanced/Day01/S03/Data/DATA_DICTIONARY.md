# Data dictionary · Geomechanical facies case study v1.1

Synthetic Permian-analog data, six horizontal wells. Roles follow the customer's dataset table: **Input** (model or calculation input), **Calibration label** (core tests, used only to calibrate log-derived rock mechanics), **Stress calibration** (DFIT closures, used only to fit the tectonic strain), **Training label** (what the model learns). Calibration and training labels are never model inputs. **Key** columns join the files.

Constants not in the files: overburden gradient 1.10 psi/ft. Join core tests to the logs on WELL and MD_FT; join stages to the logs by MD_FT between TOP_MD and BASE_MD.


## permian_analog_lateral_logs.csv

| Column | Units | Description | Role |
|---|---|---|---|
| WELL | – | Well name, PB-1H to PB-6H | Key |
| MD_FT | ft | Measured depth along the lateral, 5 ft sampling | Key |
| TVD_FT | ft | True vertical depth | Input |
| RHOB | g/cc | Bulk density | Input |
| DTC | µs/ft | Compressional slowness, dipole sonic | Input |
| DTS | µs/ft | Shear slowness, dipole sonic | Input |
| QUARTZ | fraction | Quartz, XRD-calibrated mineralogy | Input |
| CARBONATE | fraction | Calcite plus dolomite, XRD-calibrated mineralogy | Input |
| CLAY | fraction | Total clay, XRD-calibrated mineralogy | Input |
| TOC | wt% | Total organic carbon | Input |

## permian_analog_stages.csv

| Column | Units | Description | Role |
|---|---|---|---|
| WELL | – | Well name | Key |
| STAGE | – | Stage number, 1 at the shallowest (heel) end of the lateral | Key |
| TOP_MD | ft | Stage top, measured depth | Input |
| BASE_MD | ft | Stage base, measured depth | Input |
| STAGE_LEN_FT | ft | Stage length | Input |
| CLUSTERS | count | Perforation clusters in the stage | Input |
| CLUSTER_MD | ft | Cluster depths, semicolon-separated | Input |
| PROPPANT_LB_FT | lb/ft | Proppant intensity as pumped | Input |
| FLUID_BBL_FT | bbl/ft | Fluid intensity as pumped | Input |
| RATE_BPM | bbl/min | Average pump rate | Input |
| PLT_CONTRIB_PCT | % | Production-log share of well production from this stage; sums to 100 per well | Training label |
| WELL_IP_BOPD | bopd | Initial well rate (synthetic units, relative comparison only) | Training label |

## permian_analog_core_tests.csv

| Column | Units | Description | Role |
|---|---|---|---|
| WELL | – | Cored well, PB-2H or PB-6H | Key |
| MD_FT | ft | Plug depth; matches a lateral-log sample | Key |
| STATIC_YM_MPSI | Mpsi | Static Young's modulus, triaxial test | Calibration label |
| STATIC_PR | – | Static Poisson's ratio, triaxial test | Calibration label |
| UCS_PSI | psi | Unconfined compressive strength | Calibration label |

## permian_analog_dfit.csv

| Column | Units | Description | Role |
|---|---|---|---|
| WELL | – | Well name | Key |
| MD_FT | ft | Test depth, measured | Key |
| TVD_FT | ft | Test depth, true vertical | Stress calibration |
| CLOSURE_PSI | psi | Fracture closure pressure from the DFIT | Stress calibration |

## permian_analog_wells.csv

| Column | Units | Description | Role |
|---|---|---|---|
| WELL | – | Well name | Key |
| LATERAL_FT | ft | Lateral length | Input |
| TVD_FT | ft | Nominal landing true vertical depth | Input |
| PP_GRAD | psi/ft | Pore pressure gradient; needed for Shmin | Input |
| STAGE_LEN_FT | ft | Design stage length (geometric) | Input |
| CLUSTERS | count | Design clusters per stage | Input |
| PROPPANT_LB_FT | lb/ft | Design proppant intensity | Input |
| FLUID_BBL_FT | bbl/ft | Design fluid intensity | Input |
| RATE_BPM | bbl/min | Design pump rate | Input |
