from artificial_anomalies import Artificial_Anomalies
import pandas as pd
import numpy as np

# col = pd.Series([8,9,1,2,3,1,2,3,8,9, 
#                  8,9,1,2,3,1,2,3,8,9,
#                  8,9,1,2,3,1,2,3,8,9])

col = pd.Series([8,9,1,2,3,1,2,3,8,9])

col, first_iteration_ranges, reduced_ranges, ground_truth = Artificial_Anomalies.reduce_loops(col)

print('\n\nRESULTS:')
print('\ncol:')
print( col )
print('\nfirst_iteration_ranges:')
print(first_iteration_ranges)
print('\nreduced_ranges:')
print(reduced_ranges)
print('\nground_truth:')
print(ground_truth)
