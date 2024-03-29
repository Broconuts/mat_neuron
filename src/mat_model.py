import sys
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Variables for CLI.
SUPPORTED_MODEL_TYPES = ['regular_spiking', 'intrinsic_bursting', 'fast_spiking', 'chattering', "regular_spiking*"]
PROVIDED_NEURON_TYPE_DATA = ['regular_spiking', 'fast_spiking']
VALID_DELTAS = [20, 40]

# Membrane potential and spike threshold dynamics constants.
TAU_M = 50  # used to be ms
R = 50  # resistance in megaOhm
TAU_1 = 100  # used to be ms
TAU_2 = 2000  # used to be ms
PERIOD = 20  # refractory period in used to be ms

def predict(input_current: np.array, neuron_type: str, visualize: bool=False):
    """
    Predicts spikes provided an array of input currents.

    Parameters
    ---------
        input_current : np.array
            Input array of currents at each timestep.
        neuron_type : str
            Type of neuron to model. Influenes spike
            threshold dynamics variables.

    Returns
    ---------
        spike_response : list
            Binary array corresponding to input current array representing timestep of spikes.
            spike_response[t] == 1 if model proposed a spike, spike_response[t] == 0 otherwise.
    """
    # Store variables for each timestep t.
    spike_responses = []
    spikes = []
    voltage = 0
    voltages = []
    thresholds = []
    print("Predicting spiking behavior for provided timesteps...")
    # Assume that each step i represents 1ms
    for i, current in enumerate(tqdm(input_current, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
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

    if visualize:
        viz(i, voltages, thresholds, input_current, spike_responses)

    return spike_responses


def get_spike_threshold_variables(neuron_type: str):
    """
    Get spike threshold dynamic time dependent variables for modeling different types of neurons. Values derived from
    Kobayashi et al. based on optimization on simulated injected current data.

    Parameters
    ---------
        neuron_type : str
            Type of neuron to model. Influenes spike threshold dynamics variables. Can be regular_spiking,
            instrinsic_bursting, fast_spiking, or chattering.

    Returns
    ---------
        alpha_1 : float
            First time-dependent constant.
        alpha_2 : float
            Second time-dependent constant.
        w : int
            Resting value.

    """
    assert neuron_type in SUPPORTED_MODEL_TYPES

    if neuron_type == 'regular_spiking':
        return 37, 2, 19
    if neuron_type == 'regular_spiking*':
        return 200, 3, 19
    elif neuron_type == 'intrinsic_bursting':
        return 1.7, 2, 26
    elif neuron_type == 'fast_spiking':
        return 10, 0.002, 11
    elif neuron_type == 'chattering':
        return -0.5, 0.4, 26


def get_spike_threshold(t: int, spikes: list, neuron_type: str):
    """
    Determines the spike threshold at time t given the times of previous spikes.

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
    TODO: finish this!
    Get model potential for a given input current.

    Parameters
    ---------
        current : float
            Current in nA at current timestep of input.

    Returns
    ---------
        model_potential : float
            Model (membrane) potential based on input current in mV.
    """
    # Non-resetting leaky integrator (Equation 1 p. 2)
    return (R * current - voltage) / TAU_M


def generate_normal_input_currents(size: int = 100, mu: float = 0.42, sigma: float = 0.14) -> np.array:
    """
    "Randomly" generates array of size n of currents to be injected into the modelled neuron. Follows a normal
    distribution.

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


def generate_uniform_input_currents(size: int = 100, low: float = 0.1, high: float = 2.0) -> np.array:
    """
    Randomly generates array of size n of current to be injected into the neuron model. Follows a uniform distribution.

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


def get_ground_truth_input_and_response(neuron_type: str = 'regular_spiking') -> tuple:
    """
    Loads ground-truth neuron data from the QSNM Competition 2009, parses this for compatibility with the implemented
    MAT* model, and returns the input current and spike response train for the selected, available neuron type.

    Input data of QSNM Competition 2009:
        timestep per current:   in 0.1μs
        voltage:                in pA
    """
    assert neuron_type in ['fast_spiking', 'regular_spiking'], 'Neuron type not supported.'
    # dicts with locations for data separated by spiking behavior
    neuron_type_current = {'regular_spiking': "src/data/challenge_a/current.txt",
                           'fast_spiking':'src/data/challenge_b/current.txt'}
    neuron_type_voltage = {'regular_spiking': "src/data/challenge_a/voltage_allrep.txt",
                           'fast_spiking':'src/data/challenge_b/voltage_allrep.txt'}

    # load data for voltage
    with open(neuron_type_voltage[neuron_type], "r") as f:
        # start at later timestep because beginning of data is irrelevant.
        start_timestep = 150_000
        # dict with one list per trial (data contains 13 trials)
        rep = 13 if neuron_type == 'regular_spiking' else 9
        voltage = {str(k+1):[] for k in range(rep)}
        lines = f.readlines()
        # as the dataset only provides us with a smaller set of voltage values than currents, we need to make sure both
        #  variables contain the same amount of data
        end_of_data = len(lines)
        for line in lines[start_timestep:]:
            # as each line contains one value per trial, separate these and sort them into the correct list
            for i, item in enumerate(line.split("  ")):
                if i == 0: continue  # bug handling for the circumstance that every line starts with an empty item
                # voltage[str(i)].append(float(item))
                voltage[str(i)].append(float(item) >= 0 and True not in voltage[str(i)][-20:])

        global ACTUAL_SPIKETRAIN_PLOT
        ACTUAL_SPIKETRAIN_PLOT = voltage['9']

    # calculate the reliability R (averaged coincidence factor that is gathered by comparing spike trains of different
    #  repetitions)
    r_sum_component = 0
    for i in range(1, rep+1):
        current_key = str(i)
        r_sum_component += evaluate_predictions_against_ground_truth(voltage[current_key], voltage)
    r = r_sum_component / rep

    # load data for current
    with open(neuron_type_current[neuron_type], "r") as f:
        current = [line.rstrip() for line in f]
    # remove current data that we do not have voltage data for
    current = current[start_timestep:end_of_data]
    # convert current from pA to nA to be compatible with our model
    #  also: handle casting into float here, values were stored as str prior to this point
    for i, value in enumerate(current):
        current[i] = float(value) * 1e-3

    return current, voltage, r


def evaluate_predictions_against_ground_truth(prediction: list, ground_truth: dict, reliability: int = 1,
                                              delta: int = 20):
    """
    Evaluates the accuracy of model predictions against ground-truth data from the QSNM Competition 2009.

    Parameters
    ---------
        prediction : list
            Predicted spike response from model.
        ground_truth : dict
            Ground-truth predicted spike responses for each repetition in the sample data experimental trials.
        delta : int
            Allowable range of time for spikes to be considered coincident, measured in ms. The default per the source
            paper is 2ms.

    Returns
    ---------
        acc : float
            Prediction accuracy per the evaluation method stated in the challenge information.
    """
    # Get number of spikes predicted by the model.
    n_spikes_model = prediction.count(1)
    # Get the first spike to avoid calculating firing rate over empty segments.
    first_spike = prediction.index(1)
    # Calculate firing rate for determining Poisson-coincidence value.
    firing_rate = prediction[first_spike:first_spike+5000].count(1) / 5000
    coincidence_factors = []
    # Iterate through the ground-truth repetitions.
    for _, spike_train in ground_truth.items():
        # Get number of spikes in the ground-truth data.
        n_spikes_data = spike_train.count(True)
        # Calculate n expexted coincident spikes of homogeneous Poisson process.
        n_coincidence_poisson = 2 * firing_rate * n_spikes_data
        n_coincidence_model = 0
        # Iterate through timesteps of ground-truth to identify coincident
        # spikes in the prediction.
        for i, spike in enumerate(spike_train):
            if spike and 1 in prediction[i-delta:i] + prediction[i:i+delta]:
                n_coincidence_model += 1

        # Calculate the raw coincidence factor, gamma.
        coincidence_factor = (n_coincidence_model - n_coincidence_poisson) / (n_spikes_data + n_spikes_model) * \
                                (2/(1-2 * firing_rate * (delta/10)))

        # Use inter-repetition reliability score to normalize the gamma values.
        R = reliability
        coincidence_factors.append(coincidence_factor / R)

    # Gamma coincidence factor is average of normalized coincidence factor across all repetitions.
    return sum(coincidence_factors) / len(coincidence_factors)


def viz(steps: int, voltages: list, thresholds: list, input_current: np.array, spikes: list, slice_length: int = 5_000):
    """
    Creates graphs visualizing the actual firing of a recorded neuron compared to predictions of the MAT model
    given identical input currents.

    Parameters
    ----------
        steps : int
            The total number of timesteps for which the model simulated behavior.
        voltages : list
            A list of the model potentials of the MAT model at each timestep t.
        thresholds : list
            A list of the spike thresholds of the MAT model at each timestep t.
        input_current : np.array
            The input array (1D) of currents at each timestep t.
        spikes : list
            List containing information regarding whether the MAT model simulated a spike at timestep t.
            spikes[t] == 1 if model proposed a spike, spikes[t] == 0 otherwise.
        slice_length : int
            The amount of timesteps that is to be contained within a single graph.
    """
    print("Generating graphs for performance visualization...")
    # calculate amount of steps necessary
    viz_steps = steps/slice_length
    for step in tqdm(range(round(viz_steps)), bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
        fig = plt.figure()
        # make sure that y-axis has same scale for all plots for easier comparison
        plt.ylim(-20, 240)
        # plot model potential over time for current time slice
        plt.plot(range(slice_length), voltages[slice_length * step:slice_length * (step+1)], label="Potential")
        # plot spike threshold over time for current time slice
        plt.plot(range(slice_length), thresholds[slice_length * step:slice_length * (step+1)], label="Spike Threshold")
        # plot input currents over time for current time slice
        plt.plot(range(slice_length), input_current[slice_length * step:slice_length * (step+1)], label="Input Current")
        # mark timesteps when the MAT model assumed a spike
        for i, spike in enumerate(spikes[slice_length * step:slice_length * (step+1)]):
            if spike:
                plt.axvline(x=i, color='k', linestyle='--')
        # mark timesteps when a spike actually occurred in the recorded data
        for i, spike in enumerate(ACTUAL_SPIKETRAIN_PLOT[slice_length * step:slice_length * (step+1)]):
            if spike:
                plt.axvline(x=i, color='r', linestyle='--')
        plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=True)
        # save figures
        plt.savefig('images/figure' + str(step) + '.png')
        plt.close(fig)


if __name__ == '__main__':
    # retrieve command line params
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_type', help="specify the parameters of the model, determining which neuron type \
                        it should model", type=str)
    parser.add_argument('-r', '--recording_type', help="specify which neuron type data is to be loaded for evaluation",
                        type=str)
    parser.add_argument('-d', '--delta', help="determine the acceptable time-difference in spiking activity for \
                        evaluation metric calculation", type=int)
    parser.add_argument('-v', '--visualize', help="create plots of model behavior in the images directory",
                        action='store_true')
    args = parser.parse_args()

    # handle user inputs
    if vars(args)["model_type"]:
        if vars(args)["model_type"] in SUPPORTED_MODEL_TYPES:
            model_type = vars(args)["model_type"]
        else:
            print("Invalid model type. Please only chose from one of the following: ", str(SUPPORTED_MODEL_TYPES))
            print("Proceeding with default value.")
            model_type = "regular_spiking"
    else:
        model_type = "regular_spiking"
    if vars(args)["recording_type"]:
        if vars(args)["recording_type"] in PROVIDED_NEURON_TYPE_DATA:
            recording_type = vars(args)["recording_type"]
        else:
            print("Invalid recorded neuron type. Please only chose from one of the following: ",
                    str(PROVIDED_NEURON_TYPE_DATA))
            print("Proceeding with default value.")
            recording_type = "regular_spiking"
    else:
        recording_type = (model_type if model_type in PROVIDED_NEURON_TYPE_DATA else "regular_spiking")

    if vars(args)["delta"]:
        if vars(args)["delta"] in VALID_DELTAS:
            delta = vars(args)["delta"]
        else:
            print("Invalid delta value. Please chose either value 20 or 40.")
            print("Proceeding with default value.")
            delta = 20
    else:
        delta = 20
    if vars(args)["visualize"]:
        visualize = True
    else:
        visualize = False

    print("Modelled neuron type:\t\t\t\t", model_type)
    print("Using data of neuron type:\t\t\t", recording_type)
    print("Calculating coincidence factor using delta of\t", str(delta/10), "ms")

    # Set type of neuron we want to model
    neuron_type = 'regular_spiking'

    # Get ground-truth input current.
    input_current, spike_response_actual, reliability = get_ground_truth_input_and_response(recording_type)

    # Predict spikes.
    spike_response_pred = predict(input_current, model_type, visualize=visualize)

    # Evaluate predicted spikes.
    score = evaluate_predictions_against_ground_truth(spike_response_pred, spike_response_actual, reliability,
            delta=delta)

    print("Process complete!\n\nModel performed with coincidence factor: ", str(round(score, 3)))
