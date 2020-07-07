import numpy as np
from mat_model import predict, generate_input_currents


# Generate random input current.
input_current = generate_input_currents(mu=0.015, sigma=2)
# Predict spikes.
spike_response = predict(input_current)

# TODO: Add spike dynamic variables to predict method.
# TODO: Add presets for different types of neurons.
# TODO: Compare computation with other models.
