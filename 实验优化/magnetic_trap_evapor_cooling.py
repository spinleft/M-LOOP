from __future__ import absolute_import, division, print_function
__metaclass__ = type

#Imports for M-LOOP
import mloop.interfaces as mli
import mloop.controllers as mlc
import mloop.visualizations as mlv

#Other imports
import numpy as np
import time
import os
import utilities
import waveform

#Declare your custom class that inherets from the Interface class
class CustomInterface(mli.Interface):
    
    #Initialization of the interface, including this method is optional
    def __init__(self):
        #You must include the super command to call the parent class, Interface, constructor 
        super(CustomInterface,self).__init__()
        
        #Attributes of the interface can be added here
        #If you want to precalculate any variables etc. this is the place to do it
        #In this example we will just define the location of the minimum
        self.startpoint = 1
        self.endpoint = 0
        self.tf = 0.03
        self.sample_rate = 100000

        self.wave_dir = "//192.168.0.134/Share/mlparams/waveform"               # 波形文件目录
        self.tf_filename = "//192.168.0.134/Share/mlparams/waveform/tf.txt"     # 终止时间文件名
        self.signal_dir = "//192.168.0.134/Share/mlparams/index"                # 实验信号文件目录
        self.result_dir = "./results"
        self.signal_index = 1                                                   # 信号文件初始序号
        self.init_result_index = 183                                            # 初始结果序号
        self.result_index = self.init_result_index
        
    #You must include the get_next_cost_dict method in your class
    #this method is called whenever M-LOOP wants to run an experiment
    def get_next_cost_dict(self,params_dict):
        params = params_dict['params']
        # 生成波形
        wave = waveform.CubicSplineInterpolation(self.startpoint, self.endpoint, self.tf, self.sample_rate, params)
        # 保存波形到文件
        wave_filename = os.path.join(self.wave_dir, 'wave.txt')
        utilities.save_params_to_file(wave_filename, wave)
        # 发送信号文件
        signal_filename = os.path.join(
            self.signal_dir, str(self.signal_index) + '.txt')
        utilities.save_params_to_file(signal_filename, [])
        # 信号文件序号增一
        self.signal_index += 1
        # 读取实验结果
        result_filename = os.path.join(
            self.result_dir, str(self.result_index) + '.txt')
        temp = utilities.get_result_from_file(result_filename)
        # 计算cost
        bad = False
        cost = temp[0]
        if temp[1] < 1e6:
            bad = True
        # 产生结果，结果文件序号增一
        self.result_index += 1
        while bad == True:
            # 失锁等原因产生坏数据，重新进行实验
            # 保存波形到文件
            wave_filename = os.path.join(self.wave_dir, 'wave.txt')
            utilities.save_params_to_file(wave_filename, wave)
            # 发送信号文件
            signal_filename = os.path.join(
                self.signal_dir, str(self.signal_index) + '.txt')
            utilities.save_params_to_file(signal_filename, [])
            # 参数文件序号增一
            self.signal_index += 1
            # 读取实验结果
            result_filename = os.path.join(
                self.result_dir, str(self.result_index) + '.txt')
            temp = utilities.get_result_from_file(result_filename)
            # 计算cost
            cost = temp[0]
            self.result_index += 1
            # bad = ...
            if temp[1] > 1e6:
                bad = False
        print("atom num = %f, cost = %.5f"%(temp[1], temp[0]))
        cost_dict = {'cost': cost, 'bad': False, 'uncer': 0.1}
        return cost_dict
    
def main():
    #M-LOOP can be run with three commands
    
    #First create your interface
    interface = CustomInterface()
    #Next create the controller. Provide it with your interface and any options you want to set
    controller = mlc.create_controller(interface,
                                    #    controller_type='neural_net',
                                       max_num_runs = 100,
                                       target_cost = -1,
                                       num_params = 6,
                                       min_boundary = [0, 0, 0, 0, 0, 0],
                                       max_boundary = [1, 1, 1, 1, 1, 1])
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