# Neurodynamics Final Project
Implementation as well as evaluation and discussion of the multi-timescale adaptive threshold (MAT) neuron model.

## The Model
For a detailed description of the original model this implementation is based on, please refer to the original paper by Kobayashi et al. (2009). For more information on this particular implementation, please refer to our report. Both sources are included in this repository.

## Instructions
Please make sure to have all dependencies found in _requirements.txt_ installed on your system/in the virtual environment you are using. After cloning this repository, from the root of the repository's directory, call the mat_model python script.

If you want to specify parameters for testing and evaluating the model rather than using the default parameters, you can do so without changing any code. Simply specify the parameters that you want to use when calling the script. A list of options is provided when calling <pre>python src\mat_model.py -h</pre> or <pre>python src\mat_model.py --help</pre> Furthermore, the list of parameters and their potential values can be found below. Please note that the parameters can be specified in any order.

|   |  Parameter Name|                                                                                      Function|Supported Values|
|---|----------------|----------------------------------------------------------------------------------------------|----------------|
| -m|    --model type|            specify the parameters of the model, determining which neuron type it should model|regular_spiking, intrinsic_bursting, fast_spiking, chattering, regular_spiking|
| -r|--recording_type|                                 specify which neuron type data is to be loaded for evaluation|                                                 regular_spiking, fast_spiking|
| -d|         --delta|determine the acceptable time-difference in spiking activity for evaluation metric calculation|                                                                        20, 40|
| -v|     --visualize|                                        create plots of model behavior in the images directory|                                            (no value, param works on its own)|

**Examples**
<pre>python src\mat_model.py -m fast-spiking -r regular spiking -d 40 -v</pre>

_run the model using fask-spiking parameters, evaluating the performance on regular-spiking data with coincidence factor calculation using delta value 40, creating plots in the "images" directory afterwards_

### A Note on Performance
Please note that the time it takes for the model to run varies on the parameters it is run on.

## Sources
Kobayashi, R., Tsubo, Y., & Shinomoto, S. (2009). Made-to-order spiking neuronmodel  equipped  with  a  multi-timescale  adaptive  threshold.  https://www.frontiersin.org/articles/10.3389/neuro.10.009.2009/full