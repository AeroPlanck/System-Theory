# Boundary-lattice interaction-range sweep

Fixed K=40.0, N=2000, alpha/pi=0.5, v=3.0, dt=0.005, steps=50000; seeds=[9, 10, 11].

## Cell summary (formed-state wavelength statistics exclude failed runs)

| protocol          |   d0 |   diameter |   diameter_over_d0 |   realizations |   formed_count |   formed_rate | formed_modes   |   observed_mode_median |   observed_mode_min |   observed_mode_max |   effective_arc_spacing_median |   effective_arc_spacing_min |   effective_arc_spacing_max |   actual_chord_mean_median |   actual_chord_mean_min |   actual_chord_mean_max |   arc_over_d0_median |   arc_over_d0_min |   arc_over_d0_max |   actual_chord_over_d0_median |   actual_chord_over_d0_min |   actual_chord_over_d0_max |   wall_distance_over_d0_median |   wall_distance_over_d0_min |   wall_distance_over_d0_max |   fourier_locking_time_10_frames_median |   fourier_locking_time_10_frames_min |   fourier_locking_time_10_frames_max |   bulk_quantized_mode_median |   bulk_quantized_mode_min |   bulk_quantized_mode_max |   observed_minus_bulk_mode_median |   observed_minus_bulk_mode_min |   observed_minus_bulk_mode_max |
|:------------------|-----:|-----------:|-------------------:|---------------:|---------------:|--------------:|:---------------|-----------------------:|--------------------:|--------------------:|-------------------------------:|----------------------------:|----------------------------:|---------------------------:|------------------------:|------------------------:|---------------------:|------------------:|------------------:|------------------------------:|---------------------------:|---------------------------:|-------------------------------:|----------------------------:|----------------------------:|----------------------------------------:|-------------------------------------:|-------------------------------------:|-----------------------------:|--------------------------:|--------------------------:|----------------------------------:|-------------------------------:|-------------------------------:|
| fixed_diameter    | 0.75 |      4.58  |            6.10667 |              3 |              3 |      1        | 19,19,19       |                     19 |                  19 |                  19 |                       0.749771 |                    0.749405 |                    0.750772 |                   0.746364 |                0.745999 |                0.74736  |             0.999695 |          0.999206 |           1.00103 |                      0.995152 |                   0.994665 |                    0.99648 |                      0.0303128 |                   0.0262764 |                   0.0317903 |                                  125    |                                 37.5 |                                127.5 |                           16 |                        16 |                        16 |                                 3 |                              3 |                              3 |
| fixed_diameter    | 1    |      4.58  |            4.58    |              3 |              3 |      1        | 14,14,14       |                     14 |                  14 |                  14 |                       1.01602  |                    1.01584  |                    1.01621  |                   1.00752  |                1.00734  |                1.0077   |             1.01602  |          1.01584  |           1.01621 |                      1.00752  |                   1.00734  |                    1.0077  |                      0.0261265 |                   0.0257157 |                   0.0265368 |                                   50    |                                 47.5 |                                 50   |                           12 |                        12 |                        12 |                                 2 |                              2 |                              2 |
| fixed_diameter    | 1.25 |      4.58  |            3.664   |              3 |              2 |      0.666667 | 11,11          |                     11 |                  11 |                  11 |                       1.29211  |                    1.29175  |                    1.29246  |                   1.27462  |                1.27427  |                1.27496  |             1.03368  |          1.0334   |           1.03397 |                      1.01969  |                   1.01942  |                    1.01997 |                      0.0223232 |                   0.0218252 |                   0.0228212 |                                   51.25 |                                 47.5 |                                 55   |                            9 |                         9 |                         9 |                                 2 |                              2 |                              2 |
| similarity_scaled | 0.75 |      2.475 |            3.3     |              3 |              3 |      1        | 10,10,10       |                     10 |                  10 |                  10 |                       0.76239  |                    0.762368 |                    0.762487 |                   0.749917 |                0.749888 |                0.750007 |             1.01652  |          1.01649  |           1.01665 |                      0.999889 |                   0.99985  |                    1.00001 |                      0.0321574 |                   0.0319513 |                   0.0322057 |                                   32.5  |                                 25   |                                 35   |                            8 |                         8 |                         8 |                                 2 |                              2 |                              2 |
| similarity_scaled | 0.75 |      3.435 |            4.58    |              3 |              3 |      1        | 14,14,14       |                     14 |                  14 |                  14 |                       0.759503 |                    0.759464 |                    0.759615 |                   0.753208 |                0.753104 |                0.75325  |             1.01267  |          1.01262  |           1.01282 |                      1.00428  |                   1.00414  |                    1.00433 |                      0.0335977 |                   0.0332644 |                   0.0337149 |                                   45    |                                 37.5 |                                100   |                           12 |                        12 |                        12 |                                 2 |                              2 |                              2 |
| similarity_scaled | 1    |      3.3   |            3.3     |              3 |              3 |      1        | 10,10,10       |                     10 |                  10 |                  10 |                       1.02122  |                    1.02089  |                    1.02143  |                   1.00454  |                1.0042   |                1.00483  |             1.02122  |          1.02089  |           1.02143 |                      1.00454  |                   1.0042   |                    1.00483 |                      0.0246718 |                   0.0243377 |                   0.025197  |                                   40    |                                 32.5 |                                 42.5 |                            8 |                         8 |                         8 |                                 2 |                              2 |                              2 |
| similarity_scaled | 1    |      4.58  |            4.58    |              3 |              3 |      1        | 14,14,14       |                     14 |                  14 |                  14 |                       1.01602  |                    1.01584  |                    1.01621  |                   1.00752  |                1.00734  |                1.0077   |             1.01602  |          1.01584  |           1.01621 |                      1.00752  |                   1.00734  |                    1.0077  |                      0.0261265 |                   0.0257157 |                   0.0265368 |                                   50    |                                 47.5 |                                 50   |                           12 |                        12 |                        12 |                                 2 |                              2 |                              2 |
| similarity_scaled | 1.25 |      4.125 |            3.3     |              3 |              3 |      1        | 10,10,10       |                     10 |                  10 |                  10 |                       1.27961  |                    1.27932  |                    1.27969  |                   1.25867  |                1.25838  |                1.25873  |             1.02369  |          1.02345  |           1.02375 |                      1.00694  |                   1.0067   |                    1.00698 |                      0.0207508 |                   0.0206533 |                   0.0211235 |                                   47.5  |                                 40   |                                197.5 |                            8 |                         8 |                         8 |                                 2 |                              2 |                              2 |
| similarity_scaled | 1.25 |      5.725 |            4.58    |              3 |              3 |      1        | 14,14,14       |                     14 |                  14 |                  14 |                       1.27283  |                    1.27282  |                    1.27284  |                   1.26218  |                1.26217  |                1.26219  |             1.01827  |          1.01826  |           1.01827 |                      1.00974  |                   1.00973  |                    1.00975 |                      0.021132  |                   0.0211217 |                   0.0211541 |                                   50    |                                 47.5 |                                 52.5 |                           12 |                        12 |                        12 |                                 2 |                              2 |                              2 |

## Bulk one-sided spectrum

|   d0 |   bulk_k_star |   bulk_k_star_times_d0 |   bulk_lambda_star |   bulk_lambda_over_d0 |   growth_coefficient |
|-----:|--------------:|-----------------------:|-------------------:|----------------------:|---------------------:|
| 0.75 |      6.819358 |               5.114518 |           0.921375 |              1.2285   |             2.500706 |
| 1    |      5.123392 |               5.123392 |           1.226372 |              1.226372 |             2.562034 |
| 1.25 |      4.102154 |               5.127692 |           1.53168  |              1.225344 |             2.591487 |

## Descriptive scaling diagnostics

```json
{
  "inference_valid": false,
  "reason": "Three common seeds per cell; results are descriptive.",
  "similarity_scaled": {
    "formed_samples": 18,
    "actual_chord_over_d0_median": 1.0057672295520361,
    "actual_chord_over_d0_range": [
      0.99985029975895,
      1.0097493266464943
    ],
    "arc_over_d0_median": 1.0174528176755668,
    "arc_over_d0_range": [
      1.0126183694793032,
      1.0237487130600793
    ],
    "through_origin_fit_actual_chord_equals_c_d0": 1.0064559023638067,
    "through_origin_fit_arc_equals_c_d0": 1.0190597980655305,
    "actual_chord_relative_RMSE": 0.0028388436601849737,
    "arc_relative_RMSE": 0.0034411546589951775
  },
  "fixed_diameter": {
    "formed_samples": 8,
    "actual_chord_over_d0_median": 1.007425936849041,
    "actual_chord_over_d0_range": [
      0.9946649778670523,
      1.0199705885042336
    ],
    "arc_over_d0_median": 1.0159319792162886,
    "arc_over_d0_range": [
      0.9992060260657932,
      1.0339694309616005
    ],
    "through_origin_fit_actual_chord_equals_c_d0": 1.0097780663267137,
    "through_origin_fit_arc_equals_c_d0": 1.019622221397713,
    "actual_chord_relative_RMSE": 0.009380452065405376,
    "arc_relative_RMSE": 0.012984941889698093
  },
  "fixed_diameter_mode_fit": {
    "model": "m = intercept + slope/d0",
    "intercept": -1.0000000000000042,
    "slope": 15.0
  }
}
```

Three seeds per cell are sufficient for a controlled scaling screen, not for a high-precision phase-boundary or formation-probability estimate.
