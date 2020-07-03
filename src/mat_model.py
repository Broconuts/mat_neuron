import numpy as np
import math

# Membrane potential dynamics variables.
TAU_M = 5  # in ms
R = 50  # resistance in megaOhm
TAU_1 = 10  # in ms
TAU_2 = 200  # in ms

# Spike threshold dynamcs variables.
ALPHA_1 = 0
ALPHA_2 = 0
W = 0  # resting value
PERIOD = 2  # refractory period in ms



def predict(input_current, parameters):
    '''
    Predicts spikes provided an array of input currents.

    Parameters
    ---------
        input_current : np.array 
            Input array of currents at each timestep.
        parameters : 
    
    Returns
    ---------
        spike_response : np.array 
            Binary array corresponding to input current array
            representing timestep of spikes. 
    '''
    spikes = []
    for i, current in enumerate(input_current):
        # Get membrane potential.
        # Assumption: no reset of the voltage
        voltage += get_model_potential(current)
        
        # Get adaptive (spike) threshold.
        spike_threshold = get_spike_threshold(i, spikes)  

        # Check for spike.
        if voltage >= spike_threshold:  # TODO: inlcude relation to refractory period
            # Store t when there is a spike.
            spikes.append(i)

            
def get_spike_threshold(t, spikes):
    '''
    Determines the spike threshold at time t given 
    the times of previous spikes.

    Parameters
    ---------
        t : int
            Time point we want to calculate the spike threshold for.
        spikes : list
            List of time points where spikes occurred (up to time point t).
    '''
    if not spikes:
        theta = W  # if no spike has occurred yet, set theshold to resting value
    else:
        for k in spikes:
            h_t_1 += ALPHA_1 * math.exp((-t - k) / TAU_1)
            h_t_2 += ALPHA_2 * math.exp((-t - k) / TAU_2)

        theta = h_t_1 + h_t_2 + W

    return theta


def get_model_potential(current):
    '''
    Get model potential for a given input
    current.
    '''
    return -(R / TAU_M * current)


def check_spike(voltage, spike_threshold):
    '''

    '''


    return