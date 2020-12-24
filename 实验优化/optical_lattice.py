# Imports for python 2 compatibility
from __future__ import absolute_import, division, print_function

__metaclass__ = type

# Imports for M-LOOP
import mloop.interfaces as mli
import mloop.controllers as mlc
import mloop.visualizations as mlv

# Other imports
import numpy as np
import time
import io
import os
import math
import matplotlib.pyplot as plt
import waveform


def wave(params, index):
    wavepath = '波形图路径'
    filename = os.path.join(wavepath, str(index) + '.txt')
    sample_rate = 50000
    wave = waveform.CubicSplineInterpolation(startpoint=0.0, endpoint=1.0, xmax=2.0, sample_rate=sample_rate, params=params)
    np.savetxt(filename, wave, fmt="%.5f")


def readresult(index):
    resultpath = '结果路径'
    filename = os.path.join(resultpath, str(index) + '.txt')
    print('Reading result from file: ' + filename)
    while not os.path.exists(filename):
        time.sleep(2)
    time.sleep(0.5)
    result = np.loadtxt(filename)
    return result


# Declare your custom class that inherets from the Interface class
class CustomInterface(mli.Interface):

    # Initialization of the interface, including this method is optional
    def __init__(self):
        # You must include the super command to call the parent class, Interface, constructor
        super(CustomInterface, self).__init__()
        self.index = 0
        # Attributes of the interface can be added here
        # If you want to precalculate any variables etc. this is the place to do it
        # In this example we will just define the location of the minimum
        # self.minimum_params = np.array([0, 0.1, -0.1])

    # You must include the get_next_cost_dict method in your class
    # this method is called whenever M-LOOP wants to run an experiment
    def get_next_cost_dict(self, params_dict):
        # Get parameters from the provided dictionary
        params = params_dict['params']
        wave(params, self.index)
        # time.sleep(20)
        # Here you can include the code to run your experiment given a particular set of parameters
        # In this example we will just evaluate a sum of sinc functions
        result = readresult(self.index)
        psd = result[0]
        psd0 = result[1]
        atom_num = result[2]
        atom_num0 = result[3]
        uncer = result[4]

        cost = np.log(psd/psd0) / np.log(atom_num/atom_num0)

        if atom_num < 1e5:
            bad = True
        else:
            bad = False

        # The cost, uncertainty and bad boolean must all be returned as a dictionary
        # You can include other variables you want to record as well if you want
        # cost_dict = {'cost': cost, 'uncer': uncer, 'bad': bad}
        cost_dict = {'cost': cost, 'uncer': uncer, 'bad': bad}
        self.index = self.index + 1
        return cost_dict


def main():
    # M-LOOP can be run with three commands

    # First create your interface
    interface = CustomInterface()
    # Next create the controller. Provide it with your interface and any options you want to set
    controller = mlc.create_controller(interface,                                                                                                 
                                       controller_type='neural_net',
                                       max_num_runs=1000,                                                                                         
                                       target_cost=-10,
                                       num_params=8,
                                       # num_training_runs=None,                                                                                  
                                       first_params=[0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64],
                                       min_boundary=[0, 0, 0, 0, 0, 0, 0, 0],
                                       max_boundary=[1, 1, 1, 1, 1, 1, 1, 1],
                                       param_names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'],
                                       # length_scale=None,                                                                                       
                                       # trust_region=0.05,
                                       visualizations=True)                                                                                       
    # controller.ml_learner.bias_func_cycle = 4
    # controller.ml_learner.bias_func_cost_factor = [3.0,2.0,1.0,0.0]
    # controller.ml_learner.bias_func_uncer_factor = [0.0,1.0,2.0,3.0]
    # To run M-LOOP and find the optimal parameters just use the controller method optimize
    controller.optimize()

    # The results of the optimization will be saved to files and can also be accessed as attributes of the controller.
    print('Best parameters found:')
    print(controller.best_params)

    # You can also run the default sets of visualizations for the controller with one command
    mlv.show_all_default_visualizations(controller)


# Ensures main is run when this code is run as a script
if __name__ == '__main__':
    main()
