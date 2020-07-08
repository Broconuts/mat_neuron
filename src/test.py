import numpy as np
from mat_model import predict, generate_normal_input_currents, generate_uniform_input_currents


# Generate random input current.
input_current = generate_uniform_input_currents()
# Predict spikes.
spike_response = predict(input_current, 'regular_spiking')

# TODO: Compare computation with other models.
# TODO: Clean and comment.
