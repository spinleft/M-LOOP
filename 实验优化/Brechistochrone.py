from __future__ import absolute_import, division, print_function
__metaclass__ = type

#Imports for M-LOOP
import mloop.interfaces as mli
import mloop.controllers as mlc
import mloop.visualizations as mlv

#Other imports
import numpy as np
import time
import utilities

#Declare your custom class that inherets from the Interface class
class CustomInterface(mli.Interface):
    
    #Initialization of the interface, including this method is optional
    def __init__(self):
        #You must include the super command to call the parent class, Interface, constructor 
        super(CustomInterface,self).__init__()
        
        #Attributes of the interface can be added here
        #If you want to precalculate any variables etc. this is the place to do it
        #In this example we will just define the location of the minimum
        
    #You must include the get_next_cost_dict method in your class
    #this method is called whenever M-LOOP wants to run an experiment
    def get_next_cost_dict(self,params_dict):
        
        #Get parameters from the provided dictionary
        params = params_dict['params']
        startpoint = 10.
        endpoint = 0.
        xmax = 15.71
        sample_rate = 5000
        wave = utilities.waveform(startpoint, endpoint, xmax, sample_rate, params)
        #Here you can include the code to run your experiment given a particular set of parameters
        #In this example we will just evaluate a sum of sinc functions
        k = 5.0
        g = 9.8
        x_step = 1.0 / sample_rate
        x = np.arange(0, xmax, x_step)
        len_x = len(x)
        t = 0
        bad = False
        for i in range(1, len_x):
            if wave[i] > 10.0 or wave[i] == float('nan'):
                bad = True
                break
            else:
                v_i = np.sqrt(2 * g * (10.0 - wave[i - 1]))
                s = np.sqrt((x[i] - x[i - 1]) ** 2 +
                            (wave[i - 1] - wave[i]) ** 2)
                a = (wave[i - 1] - wave[i]) / s
                if np.abs(a) < 1e-15:
                    t += s / v_i
                else:
                    t += (np.sqrt(v_i ** 2 + 2 * a * s) - v_i) / a
        min_time = np.pi * np.sqrt(k / g)
        if bad:
            cost_dict = {'bad':True}
        else:
            t += t * np.random.normal(0, 0.1)
            cost = t - min_time
            uncer = 0.1 * t
            cost_dict = {'cost':cost, 'uncer':uncer, 'bad':False}
        return cost_dict
    
def main():
    #M-LOOP can be run with three commands
    
    #First create your interface
    interface = CustomInterface()
    #Next create the controller. Provide it with your interface and any options you want to set
    controller = mlc.create_controller(interface,
                                    #    controller_type='neural_net',
                                       max_num_runs = 1000,
                                       target_cost = -0.5,
                                       num_params = 7, 
                                       min_boundary = [-3, -3, -3, -3, -3, -3, -3],
                                       max_boundary = [3, 3, 3, 3, 3, 3, 3])
    #To run M-LOOP and find the optimal parameters just use the controller method optimize
    controller.optimize()
    
    #The results of the optimization will be saved to files and can also be accessed as attributes of the controller.
    print('Best parameters found:')
    print(controller.best_params)
    
    #You can also run the default sets of visualizations for the controller with one command
    mlv.show_all_default_visualizations(controller)
    

#Ensures main is run when this code is run as a script
if __name__ == '__main__':
    main()