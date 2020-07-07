import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


# Membrane potential dynamics variables.
TAU_M = 5  # ms
R = 50  # resistance in megaOhm
TAU_1 = 10  # ms
TAU_2 = 200  # ms

# Spike threshold dynamics variables.
ALPHA_1 = 37 # mV
ALPHA_2 = 2 # mV
W = 19  # resting value in mV
PERIOD = 2  # refractory period in ms


def predict(input_current):
    """
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
    """
    # Store variables for each timestep t.
    spike_responses = []
    spikes = []
    voltage = 0
    voltages = []
    thresholds = []
    # Assume that each step i represents 1ms
    for i, current in enumerate(input_current):
        # Get membrane potential.
        voltage += get_model_potential(current, voltage)
        print(voltage)
        voltages.append(voltage)
        # Get adaptive (spike) threshold.
        spike_threshold = get_spike_threshold(i, spikes)
        thresholds.append(spike_threshold)
        # Check if neuron is in refractory period, according to
        # adaptive threshold MAT rule (p. 2)
        in_refractory_period = (i - spikes[-1]) <= PERIOD if spikes else False
        # Check for spike.
        if not in_refractory_period and voltage >= spike_threshold:
            # Store t when there is a spike.
            spikes.append(i)
            spike_responses.append(1)
            # Reset voltage to 0 (even though this is not assumed in the model, cf p. 2)
            # voltage = 0
        else:
            spike_responses.append(0)

    # Visualize model states.
    plt.style.use('seaborn-darkgrid')
    plt.plot(range(i+1), voltages, label="Potential")
    plt.plot(range(i+1), thresholds, label="Spike Threshold")
    plt.plot(range(i+1), input_current, label="Input Current")
    plt.legend()
    plt.savefig('figure.png')


    return spike_responses


def get_spike_threshold(t, spikes):
    """
    Determines the spike threshold at time t given
    the times of previous spikes.

    Parameters
    ---------
        t : int
            Time point we want to calculate the spike threshold for.
        spikes : list
            List of time points where spikes occurred (up to time point t).

    Returns
    ---------
        theta : float
            Spike threshold at time t.
    """
    h_t_1 = 0
    h_t_2 = 0
    # Summation over previous spikes (Equation 3 p. 2)
    for k in spikes:
        h_t_1 += ALPHA_1 * math.exp(-(t - k) / TAU_1)
        h_t_2 += ALPHA_2 * math.exp(-(t - k) / TAU_2)

    # Adaptive spike threshold (Equation 2 p. 2)
    theta = h_t_1 + h_t_2 + W

    return theta


def get_model_potential(current, voltage):
    """
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
    """
    # Non-resetting leaky integrator (Equation 1 p. 2)
    return (R * current - voltage) / TAU_M


def generate_input_currents(size: int = 100, mu: float = 15.00, sigma: float = 7.00) -> np.array:
    """
    "Randomly" generates array of size n of currents to be injected into the
    modelled neuron.

    Parameters
    -----------
        size : int
            The amount of generated input currents.
        mu : float
            The mean of the distribution the values are to be drawn from.
        sigma : float
            The standard deviation of the distribution the values are to be drawn from.
    
    Returns
    -------
        An np.array of simulated input currents.
    """
    return np.random.normal(mu, sigma, size)
 

def evaluate_predicted_spikes():
    """
    """
    pass
