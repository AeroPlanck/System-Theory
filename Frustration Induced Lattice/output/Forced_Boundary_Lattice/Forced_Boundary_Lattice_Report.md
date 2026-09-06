# Forced boundary-lattice intervention

> **Scientific status:** These are planted/intervention trajectories, not evidence of spontaneous crystallization. The first frame is deliberately rearranged; all later frames follow the unmodified model dynamics.

N=2000, dt=0.005, steps=50000, snap=100; target mode `floor(pi * D / d0)`.

## Terminal diagnostics

| family           |   strength_k |   d0 |   diameter |   seed |   target_mode |   observed_mode_terminal |   target_mode_fraction_tail | lattice_formed_terminal   | target_mode_retained   |   fourier_amplitude_terminal |   temporal_mode_stability |   shell_particle_fraction |   actual_chord_over_d0 |
|:-----------------|-------------:|-----:|-----------:|-------:|--------------:|-------------------------:|----------------------------:|:--------------------------|:-----------------------|-----------------------------:|--------------------------:|--------------------------:|-----------------------:|
| critical_failure |        20.75 | 1    |       3    |     10 |             9 |                        9 |                           1 | True                      | True                   |                     0.997694 |                         1 |                         1 |               0.993965 |
| critical_failure |        20.75 | 1    |       3.5  |     10 |            10 |                       10 |                           1 | True                      | True                   |                     0.993653 |                         1 |                         1 |               1.04351  |
| d0_failure       |        40    | 1.25 |       4.58 |      9 |            11 |                       11 |                           1 | True                      | True                   |                     0.998817 |                         1 |                         1 |               1.01936  |
| critical_failure |        20.75 | 1    |       4    |     10 |            12 |                       12 |                           1 | True                      | True                   |                     0.998326 |                         1 |                         1 |               1.00868  |
| critical_failure |        20.75 | 1    |       4.5  |     11 |            14 |                       14 |                           1 | True                      | True                   |                     0.995661 |                         1 |                         1 |               0.974967 |
| critical_failure |        20.75 | 1    |       5    |     11 |            15 |                       15 |                           1 | True                      | True                   |                     0.99884  |                         1 |                         1 |               1.01809  |

## Provenance

Each HDF5 file contains `/positionX`, `/phaseTheta`, and `/metadata`. The metadata records the original failed HDF5 path and labels the output as an intervention trajectory. Original HDF5 files are untouched.
