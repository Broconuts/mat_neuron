import numpy as np
import math
from mat_model import predict, get_spike_threshold


input_current = np.zeros(2000)
input_current[170:1999] = 0.55

# Spike threshold dynamcs variables.
# ALPHA_1 = 30
# ALPHA_2 = 2
# W = 20  # resting value
# PERIOD = 2  # refractory period in ms


spike_response = predict(input_current)
print(spike_response)

# print(get_spike_threshold(0, []))

# Todo: degree of coincidence evaluation
