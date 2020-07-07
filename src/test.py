import numpy as np
from mat_model import predict, generate_input_currents


# Spike threshold dynamics variables.
ALPHA_1 = 37
ALPHA_2 = 2
W = 19  # resting value
PERIOD = 2  # refractory period in ms

# Fake input current.
# input_current = np.zeros(1000)
# input_current[100] = 0.055
# input_current[150] = 0.055
# input_current[170] = 0.055
# input_current[300] = 0.055

input_current = generate_input_currents()
# input_current = np.random.uniform(low=0.0, high=0.055, size=1000)
print(input_current)
spike_response = predict(input_current)

# TODO: Evaluate predictions.
# TODO: Integrate real data.
