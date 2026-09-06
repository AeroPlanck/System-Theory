# Extreme planted boundary-mode stability

> These are intervention trajectories. They test nonlinear stability, not spontaneous accessibility.

Fixed K=20.75, alpha/pi=0.5, D=4.5, d0=1.0, N=2000, steps=50000; geometric reference m0=14.

|   target_mode |   geometric_mode |   mode_ratio_to_geometric |   initial_chord_over_d0 |   target_mode_fraction_all_frames |   first_target_departure_time |   observed_mode_terminal |   temporal_mode_stability_tail |   target_amplitude_min |   target_amplitude_terminal |   shell_fraction_min | lattice_formed_terminal   | target_mode_retained_terminal   |   actual_chord_over_d0_terminal |
|--------------:|-----------------:|--------------------------:|------------------------:|----------------------------------:|------------------------------:|-------------------------:|-------------------------------:|-----------------------:|----------------------------:|---------------------:|:--------------------------|:--------------------------------|--------------------------------:|
|             5 |               14 |                  0.357143 |                2.60389  |                         0.197605  |                          49.5 |                       13 |                              1 |             0.00208358 |                   0.0325121 |               0.8965 | False                     | False                           |                      nan        |
|             7 |               14 |                  0.5      |                1.9221   |                         0.219561  |                          55   |                       14 |                              1 |             0.00747123 |                   0.0718018 |               0.9595 | False                     | False                           |                        0.980734 |
|            21 |               14 |                  1.5      |                0.660257 |                         0.0499002 |                          12.5 |                       14 |                              1 |             0.0171863  |                   0.0429428 |               0.9815 | False                     | False                           |                        0.978609 |
|            28 |               14 |                  2        |                0.496003 |                         0.0918164 |                          23   |                       14 |                              1 |             0.0677859  |                   0.688573  |               0.98   | True                      | False                           |                        0.981706 |

The planted states include 8% cluster-center angular jitter and 0.08 rad heading noise.
