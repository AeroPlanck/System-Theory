# Critical boundary lattice: controlled K sweep

Fixed parameters: N=2000, alpha/pi=0.5, d0=1.0, v=3.0, dt=0.005, steps=50000 (t=250); seeds=[9, 10, 11].

The strict crystallization rule is identical to the earlier quantization analysis: terminal and last-20-frame median Fourier amplitudes >=0.90, mode stability >=0.90, boundary-shell fraction >=0.70, and Fourier/peak/DBSCAN counts all agree.

## Cell summary

|   strength_k |   diameter |   realizations |   lattice_formed_count |   lattice_formed_rate | formed_modes   |   observed_mode_median |   observed_mode_min |   observed_mode_max |   effective_wavenumber_median |   effective_wavenumber_min |   effective_wavenumber_max |   effective_arc_spacing_median |   effective_arc_spacing_min |   effective_arc_spacing_max |   geometric_chord_spacing_median |   geometric_chord_spacing_min |   geometric_chord_spacing_max |   actual_chord_mean_median |   actual_chord_mean_min |   actual_chord_mean_max |   wall_distance_of_clusters_median |   wall_distance_of_clusters_min |   wall_distance_of_clusters_max |   fourier_locking_time_10_frames_median |   fourier_locking_time_10_frames_min |   fourier_locking_time_10_frames_max |   fourier_amplitude_terminal_median |   fourier_amplitude_terminal_min |   fourier_amplitude_terminal_max |
|-------------:|-----------:|---------------:|-----------------------:|----------------------:|:---------------|-----------------------:|--------------------:|--------------------:|------------------------------:|---------------------------:|---------------------------:|-------------------------------:|----------------------------:|----------------------------:|---------------------------------:|------------------------------:|------------------------------:|---------------------------:|------------------------:|------------------------:|-----------------------------------:|--------------------------------:|--------------------------------:|----------------------------------------:|-------------------------------------:|-------------------------------------:|------------------------------------:|---------------------------------:|---------------------------------:|
|         8    |       3.3  |              3 |                      3 |               1       | 9,9,9          |                      9 |                   9 |                   9 |                        5.9245 |                     5.916  |                     5.9364 |                         1.0605 |                      1.0584 |                      1.0621 |                          1.0391  |                       1.0371  |                       1.0406  |                    1.0392  |                 1.0371  |                 1.0407  |                           0.13089  |                        0.1287   |                        0.13393  |                                  125    |                                100   |                                155   |                             0.98417 |                          0.98285 |                          0.98485 |
|         8    |       4.58 |              3 |                      1 |               0.33333 | 13             |                     13 |                  13 |                  13 |                        6.0036 |                     6.0036 |                     6.0036 |                         1.0466 |                      1.0466 |                      1.0466 |                          1.0364  |                       1.0364  |                       1.0364  |                    1.0364  |                 1.0364  |                 1.0364  |                           0.12462  |                        0.12462  |                        0.12462  |                                  155    |                                155   |                                155   |                             0.96656 |                          0.96656 |                          0.96656 |
|        12    |       3.3  |              3 |                      2 |               0.66667 | 9,9            |                      9 |                   9 |                   9 |                        5.8052 |                     5.8042 |                     5.8062 |                         1.0823 |                      1.0821 |                      1.0825 |                          1.0605  |                       1.0603  |                       1.0607  |                    1.0605  |                 1.0602  |                 1.0607  |                           0.099674 |                        0.099409 |                        0.099939 |                                   83.75 |                                 70   |                                 97.5 |                             0.97394 |                          0.96501 |                          0.98287 |
|        12    |       4.58 |              3 |                      2 |               0.66667 | 13,13          |                     13 |                  13 |                  13 |                        5.9479 |                     5.9467 |                     5.9492 |                         1.0564 |                      1.0561 |                      1.0566 |                          1.0461  |                       1.0459  |                       1.0463  |                    1.0461  |                 1.0459  |                 1.0463  |                           0.10437  |                        0.10391  |                        0.10482  |                                  127.5  |                                122.5 |                                132.5 |                             0.98382 |                          0.98074 |                          0.98691 |
|        20.75 |       3.3  |              3 |                      2 |               0.66667 | 10,10          |                     10 |                  10 |                  10 |                        6.2263 |                     6.2236 |                     6.229  |                         1.0091 |                      1.0087 |                      1.0096 |                          0.99262 |                       0.99219 |                       0.99305 |                    0.99263 |                 0.99222 |                 0.99304 |                           0.043904 |                        0.043212 |                        0.044596 |                                   58.75 |                                 55   |                                 62.5 |                             0.99433 |                          0.99329 |                          0.99537 |
|        20.75 |       4.58 |              3 |                      1 |               0.33333 | 14             |                     14 |                  14 |                  14 |                        6.2392 |                     6.2392 |                     6.2392 |                         1.007  |                      1.007  |                      1.007  |                          0.99862 |                       0.99862 |                       0.99862 |                    0.99861 |                 0.99861 |                 0.99861 |                           0.046127 |                        0.046127 |                        0.046127 |                                  147.5  |                                147.5 |                                147.5 |                             0.98602 |                          0.98602 |                          0.98602 |
|        40    |       3.3  |              3 |                      3 |               1       | 10,10,10       |                     10 |                  10 |                  10 |                        6.1526 |                     6.1513 |                     6.1546 |                         1.0212 |                      1.0209 |                      1.0214 |                          1.0045  |                       1.0042  |                       1.0047  |                    1.0045  |                 1.0042  |                 1.0048  |                           0.024672 |                        0.024338 |                        0.025197 |                                   40    |                                 32.5 |                                 42.5 |                             0.99844 |                          0.99841 |                          0.99858 |
|        40    |       4.58 |              3 |                      3 |               1       | 14,14,14       |                     14 |                  14 |                  14 |                        6.1841 |                     6.183  |                     6.1852 |                         1.016  |                      1.0158 |                      1.0162 |                          1.0075  |                       1.0073  |                       1.0077  |                    1.0075  |                 1.0073  |                 1.0077  |                           0.026127 |                        0.025716 |                        0.026537 |                                   50    |                                 47.5 |                                 50   |                             0.99635 |                          0.99615 |                          0.99757 |

## One-sided bulk spectrum

|   strength_k |   one_sided_k_star |   one_sided_lambda_star |   growth_coefficient |   bare_turning_length_v_over_k |   bulk_mode_D3.30 |   bulk_mode_D4.58 |
|-------------:|-------------------:|------------------------:|---------------------:|-------------------------------:|------------------:|------------------:|
|         8    |           5.292991 |                1.187076 |            0.4481133 |                      0.375     |                 9 |                12 |
|        12    |           5.043657 |                1.24576  |            0.5871964 |                      0.25      |                 8 |                12 |
|        20.75 |           5.094671 |                1.233286 |            1.224537  |                      0.1445783 |                 8 |                12 |
|        40    |           5.123384 |                1.226374 |            2.562034  |                      0.075     |                 8 |                12 |

## Statistical comparison

```json
{
  "available": true,
  "inference_valid": false,
  "reason": "Only three common seeds per K x D cell were run. Coefficients are descriptive; no p-values or bootstrap confidence intervals are reported.",
  "descriptive_model": "a_eff = intercept + slope / K + diameter fixed effect",
  "intercept": 1.0082078110109,
  "slope": 0.4844454824905884,
  "diameter_fixed_effect_offset": -0.006287182059099441,
  "reference_diameter": 3.3,
  "constant_K_model_SSE": 0.010498348291967666,
  "inverse_K_model_SSE": 0.004498762688262362,
  "SSE_reduction": 0.0059995856037053045,
  "k_values": [
    8.0,
    12.0,
    20.75,
    40.0
  ],
  "formed_sample_size": 17,
  "per_K_descriptive_ranges": {
    "8": {
      "n_formed": 4,
      "arc_spacing_median": 1.0594766354249723,
      "arc_spacing_min": 1.0465747484926606,
      "arc_spacing_max": 1.0620653703698264,
      "actual_chord_median": 1.038138573072476,
      "effective_q_median": 5.930467347032803
    },
    "12": {
      "n_formed": 4,
      "arc_spacing_median": 1.0693647878010855,
      "arc_spacing_min": 1.0561431303542665,
      "arc_spacing_max": 1.0825165973126396,
      "actual_chord_median": 1.0532907257363004,
      "effective_q_median": 5.876463428723936
    },
    "20.75": {
      "n_formed": 3,
      "arc_spacing_median": 1.0087052212973275,
      "arc_spacing_min": 1.0070477251453864,
      "arc_spacing_max": 1.0095747999982916,
      "actual_chord_median": 0.9930437991263888,
      "effective_q_median": 6.228960824747772
    },
    "40": {
      "n_formed": 6,
      "arc_spacing_median": 1.0185511295231984,
      "arc_spacing_min": 1.0158399139870942,
      "arc_spacing_max": 1.0214337314437192,
      "actual_chord_median": 1.0060839970459612,
      "effective_q_median": 6.168780696611961
    }
  }
}
```

## Formed-state medians by K

|   strength_k |   effective_arc_spacing |   actual_chord_mean |   effective_wavenumber |
|-------------:|------------------------:|--------------------:|-----------------------:|
|         8    |                 1.05948 |            1.03814  |                5.93047 |
|        12    |                 1.06936 |            1.05329  |                5.87646 |
|        20.75 |                 1.00871 |            0.993044 |                6.22896 |
|        40    |                 1.01855 |            1.00608  |                6.16878 |

Interpretation must condition on lattice formation: failed runs are kinetics/attractor outcomes, not wavelength samples.

Reference files were read-only and were not modified. Their SHA-256 values are stored in K_Sweep_Configuration.json.
