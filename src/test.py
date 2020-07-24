import numpy as np
import mat_model as mm


NEURON_TYPE = 'regular_spiking'

# Generate random input current.
# input_current = mm.generate_uniform_input_currents()

# Get ground-truth input current. 
input_current, spike_response_actual, reliability = mm.get_ground_truth_input_and_response(NEURON_TYPE)

# Predict spikes.
spike_response_pred = mm.predict(input_current, NEURON_TYPE)

print(len(spike_response_pred))

# Evaluate predicted spikes.
score = mm.evaluate_predictions_against_ground_truth(spike_response_pred, spike_response_actual, delta=15000)

print(score)

# TODO: Clean and comment.
 