import numpy as np
from mat_model import predict, generate_input_currents


# Spike threshold dynamics variables.
ALPHA_1 = 37
ALPHA_2 = 2
W = 19  # resting value
PERIOD = 2  # refractory period in ms

# Generate random input current.
input_current = generate_input_currents(mu=0.015, sigma=2)
# Predict spikes.
spike_response = predict(input_current)

# TODO: Compare computation with other models.
