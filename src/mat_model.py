import numpy as np
import math
import matplotlib.pyplot as plt


# Membrane potential and spike threshold dynamics constants.
TAU_M = 5  # ms
R = 50  # resistance in megaOhm
TAU_1 = 10  # ms
TAU_2 = 200  # ms
PERIOD = 2  # refractory period in ms

def predict(input_current, neuron_type):
    """
    Predicts spikes provided an array of input currents.

    Todo: evaluate if current assumption of i equals 1ms in real time is correct.

    Parameters
    ---------
        input_current : np.array
            Input array of currents at each timestep.
        neuron_type : str
            Type of neuron to model. Influenes spike
            threshold dynamics variables.

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
        voltages.append(voltage)
        # Get adaptive (spike) threshold.
        spike_threshold = get_spike_threshold(i, spikes, neuron_type)
        thresholds.append(spike_threshold)
        # Check if neuron is in refractory period, according to
        # adaptive threshold MAT rule (p. 2)
        in_refractory_period = (i - spikes[-1]) <= PERIOD if spikes else False
        # Check for spike.
        if not in_refractory_period and voltage >= spike_threshold:
            # Store t when there is a spike.
            spikes.append(i)
            spike_responses.append(1)
            # Voltage does not reset. cf p. 2)
        else:
            spike_responses.append(0)

    # Visualize model states.
    plt.style.use('seaborn-darkgrid')
    plt.plot(range(i+1), voltages, label="Potential")
    plt.plot(range(i+1), thresholds, label="Spike Threshold")
    plt.plot(range(i+1), input_current, label="Input Current")
    # Visualize spikes.
    for spike in spikes:
        plt.axvline(x=spike, color='k', linestyle='--')

    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=3, fancybox=True, shadow=True)
    plt.savefig('figure.png')


    return spike_responses


def get_spike_threshold_variables(neuron_type):
    """
    Get spike threshold dynamic time dependent variables
    for modeling different types of neurons. Values
    derived from Kobayashi et al. based on optimization
    on simulated injected current data.

    Parameters
    ---------
        neuron_type : str
            Type of neuron to model. Influenes spike
            threshold dynamics variables. Can be
            regular_spiking, instrinsic_bursting,
            fast_spiking, or chattering.

    Returns
    ---------
        alpha_1 : float
            First time-dependent constant.
        alpha_2 : float
            Second time-dependent constant.
        w : int
            Resting value.

    """
    assert neuron_type in ['regular_spiking',
                           'intrinsic_bursting',
                           'fast_spiking',
                           'chattering']

    if neuron_type == 'regular_spiking':
        return 37, 2, 19
    elif neuron_type == 'intrinsic_bursting':
        return 1.7, 2, 26
    elif neuron_type == 'fast_spiking':
        return 10, 0.002, 11
    elif neuron_type == 'chattering':
        return -0.5, 0.4, 26


def get_spike_threshold(t, spikes, neuron_type):
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
    # Get spike threshold variables depending on neuron type.
    alpha_1, alpha_2, w = get_spike_threshold_variables(neuron_type)
    h_t_1 = 0
    h_t_2 = 0
    # Summation over previous spikes (Equation 3 p. 2)
    for k in spikes:
        h_t_1 += alpha_1 * math.exp(-(t - k) / TAU_1)
        h_t_2 += alpha_2 * math.exp(-(t - k) / TAU_2)

    # Adaptive spike threshold (Equation 2 p. 2)
    theta = h_t_1 + h_t_2 + w

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


def generate_normal_input_currents(size: int = 100, mu: float = 0.42, sigma: float = 0.14) -> np.array:
    """
    "Randomly" generates array of size n of currents to be injected into the
    modelled neuron. Follows a normal distribution.

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


def generate_uniform_input_currents(size:int=100, low:float=0.1, high:float=2.0) -> np.array:
    """
    Randomly generates array of size n of current to be injected into the
    neuron model. Follows a uniform distribution

    Parameters
    ---------
        size : int
            The amount of generated input currents. (mV)
        low : float
            Minimum value. (mV)
        high : float
            Maximum value (mV)

    Returns
        An np.array of simulated input currents.
    """
    return np.random.uniform(low=low, high=high, size=size)

