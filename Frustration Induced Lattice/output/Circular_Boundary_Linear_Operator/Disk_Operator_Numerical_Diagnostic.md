# Normalized disk-operator numerical diagnostic

The operator is the three-field circular closure derived in Circular_Boundary_Matrix_Derivation.md, with neighbor-area normalization and the two specular moment boundary conditions.

## Resolution test

|   strength_k |   diameter | orders         | selected_m_by_order   | growth_by_order                                                       | edge_fraction_by_order                       | mode_converged   |   growth_last_relative_change |
|-------------:|-----------:|:---------------|:----------------------|:----------------------------------------------------------------------|:---------------------------------------------|:-----------------|------------------------------:|
|         8    |       3.3  | 40,50,60,70,80 | 13,6,5,9,8            | 2.5160927e-05,1.6417003e-05,7.3499494e-05,1.3025439e-05,1.2351431e-06 | 0.591149,0.515278,0.507148,0.509258,0.556514 | False            |                     9.545693  |
|         8    |       4.58 | 40,50,60,70,80 | 16,12,14,9,10         | 4.1590326e-06,2.4627998e-06,2.2780452e-06,1.9646505e-07,1.4500013e-06 | 0.609448,0.591738,0.626072,0.577031,0.502134 | False            |                     0.864507  |
|        12    |       3.3  | 40,50,60,70,80 | 13,12,5,9,10          | 7.706814e-05,4.6747513e-06,0.00016529667,3.6392369e-05,2.0031205e-05  | 0.600418,0.60692,0.50219,0.527026,0.506835   | False            |                     0.8167838 |
|        12    |       4.58 | 40,50,60,70,80 | 16,10,14,9,10         | 1.0655061e-05,1.1239971e-05,7.1029572e-06,6.1950836e-07,4.6558868e-06 | 0.634626,0.511131,0.61367,0.564614,0.501486  | False            |                     0.8669408 |
|        20.75 |       3.3  | 40,50,60,70,80 | 13,12,9,9,8           | 0.00029342913,1.8749118e-05,3.1099765e-05,9.109648e-05,1.9842261e-05  | 0.637169,0.599141,0.646054,0.564068,0.559869 | False            |                     3.591033  |
|        20.75 |       4.58 | 40,50,60,70,80 | 13,12,14,9,10         | 3.6337145e-08,4.1398844e-05,2.6503185e-05,2.5602683e-06,1.9602139e-05 | 0.695691,0.582274,0.593582,0.524929,0.52343  | False            |                     0.8693883 |
|        40    |       3.3  | 40,50,60,70,80 | 13,14,7,8,6           | 0.00068627049,0.00022949268,0.00011939704,0.00013418761,0.00010274492 | 0.727525,0.505742,0.610204,0.582601,0.581344 | False            |                     0.3060267 |
|        40    |       4.58 | 40,50,60,70,80 | 12,12,17,8,15         | 0.00043978583,0.00012447375,0.048574278,0.0005560027,4.0953405e-06    | 0.547554,0.523007,0.99957,0.547975,0.581196  | False            |                   134.7647    |

## Verdict

Converged cells: 0/8. The selected m changes with radial resolution in every tested K x D cell, while the candidate real growth is generally near zero and also fails to converge. Isolated larger growth values disappear at the next resolution and are classified as collocation/closure pseudospectral artifacts.

The particle observations are D=3.3: 9 at low K; 10 at high K; D=4.58: 13 at low K; 14 at high K. The present homogeneous three-field disk closure therefore does not provide a numerically converged prediction that can be matched to these modes.

This is a negative but informative result: circular geometry and neighbor normalization alone do not cure the alpha=pi/2 marginal degeneracy at this closure level. The next controlled calculation is linearization about the nonuniform circulating boundary base state, or a higher-harmonic kinetic wall calculation.
