import numpy as np
import math
import matplotlib.pyplot as plt

# Membrane potential dynamics variables.
TAU_M = 5  # in ms
R = 50  # resistance in megaOhm
TAU_1 = 10  # in ms
TAU_2 = 200  # in ms

# Spike threshold dynamcs variables.
ALPHA_1 = 37 #  mV
ALPHA_2 = 2 #  mV
W = 19  # resting value mV
PERIOD = 2  # refractory period in ms


def predict(input_current):
    '''
    Predicts spikes provided an array of input currents.
    
    Todo: evaluate if current assumption of i equals 1ms in real time is correct.

    Parameters
    ---------
        input_current : np.array 
            Input array of currents at each timestep.
    
    Returns
    ---------
        spike_response : np.array 
            Binary array corresponding to input current array
            representing timestep of spikes. 
    '''
    spike_responses = []
    spikes = [] 
    voltage = 0
    thetas = []
    # TODO: Assume that each step represents 1ms
    for i, current in enumerate(input_current):    
        print("Current step: %s" % i)    
        # Get membrane potential.
        # Assumption: no reset of the voltage
        # voltage += get_model_potential(current)
        voltage += get_model_potential(current, voltage)
        
        # Get adaptive (spike) threshold.
        spike_threshold = get_spike_threshold(i, spikes)
        thetas.append(spike_threshold)

        # check if neuron is in refractory period
        # TODO: investigate scaling the refractory period
        if spikes:
            in_refractory_period = (i - spikes[-1]) <= PERIOD
        else:
            in_refractory_period = False
        # Check for spike.
        if not in_refractory_period and voltage >= spike_threshold:  # TODO: inlcude relation to refractory period
            # Store t when there is a spike.
            spikes.append(i)
            spike_responses.append(True)
        else:
            spike_responses.append(False)

    fig = plt.plot(range(i+1), thetas)
    plt.show()
    
    return spike_responses
            
            
def get_spike_threshold(t, spikes, plot=False):
    '''
    Determines the spike threshold at time t given 
    the times of previous spikes.

    Parameters
    ---------
        t : int
            Time point we want to calculate the spike threshold for.
        spikes : list
            List of time points where spikes occurred (up to time point t).
        plot : boolean
            Create plot of values over time.
            
    Returns
    ---------
        theta : float
            Spike threshold at time t.
    '''
    if not spikes:
        theta = W  # if no spike has occurred yet, set threshold to resting value
        
    else:
        if plot:
            ht1 = []
            ht2 = []
            theta_list = []
        h_t_1 = 0
        h_t_2 = 0
        for k in spikes:
            # print(t-k)
            h_t_1 += ALPHA_1 * math.exp(-(t - k) / TAU_1)
            h_t_2 += ALPHA_2 * math.exp(-(t - k) / TAU_2)
            if plot:
                ht1.append(h_t_1)
                ht2.append(h_t_2)
                theta_list.append(h_t_1 + h_t_2 + W)

            # h_t_1 += (ALPHA_1 * math.exp(-t / TAU_1)) - (ALPHA_1 * math.exp(- k / TAU_1))
            # h_t_2 += (ALPHA_2 * math.exp(-t / TAU_2)) - (ALPHA_2 * math.exp(- k / TAU_2))
            
        # print(t - k)
        # print(h_t_1, h_t_2)
        theta = h_t_1 + h_t_2 + W

    if plot:
        x = range(t)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='col')
        ax1.plot(x, ht1)
        ax2.plot(x, ht2)
        ax3.plot(x, theta_list)
        ax1.set_title('ht1')
        ax2.set_title('ht2')
        ax3.set_title('theta')
        plt.show()

    print("Threshold " + str(theta) + "\n")
    return theta


def get_model_potential(current, voltage):
    '''
    Get model potential for a given input
    current.

    Parameters
    ---------
        current : float
            Current in nA at current timestep of
            input.

    Returns
    ---------
        model_potential : float
            Model (membrane) potential based 
            on input current in mV.
    '''
    # Non-resetting leaky integrator.
    # TODO: Assumption that voltage should be positive?
    # potential = R / TAU_M * current
    potential = (-voltage + R * current) / TAU_M
    print("Potential " + str(potential))
    return potential
