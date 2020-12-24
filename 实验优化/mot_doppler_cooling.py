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

#Declare your custom class that inherets from the Interface class
class CustomInterface(mli.Interface):
    
    #Initialization of the interface, including this method is optional
    def __init__(self):
        #You must include the super command to call the parent class, Interface, constructor 
        super(CustomInterface,self).__init__()
        
        #Attributes of the interface can be added here
        #If you want to precalculate any variables etc. this is the place to do it
        #In this example we will just define the location of the minimum
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
        params_dict['params']
        # 生成波形
        wave1 = utilities.waveform_linear(params[0], params[1], self.tf, self.sample_rate)
        wave2 = utilities.waveform_linear(params[2], params[3], self.tf, self.sample_rate)
        # wave3 = utilities.waveform_linear(params[4], params[5], self.tf, self.sample_rate)
        # wave4 = utilities.waveform_linear(params[0], params[7], self.tf, self.sample_rate)
        # 保存波形到文件
        wave1_filename = os.path.join(self.wave_dir, '1.txt')
        wave2_filename = os.path.join(self.wave_dir, '2.txt')
        # wave3_filename = os.path.join(self.wave_dir, '3.txt')
        # wave4_filename = os.path.join(self.wave_dir, '4.txt')
        const3_filename = os.path.join(self.wave_dir, '3.txt')
        const4_filename = os.path.join(self.wave_dir, '4.txt')
        const5_filename = os.path.join(self.wave_dir, '5.txt')
        const6_filename = os.path.join(self.wave_dir, '6.txt')

        utilities.save_params_to_file(wave1_filename, wave1)
        utilities.save_params_to_file(wave2_filename, wave2)
        # utilities.save_params_to_file(wave3_filename, wave3)
        # utilities.save_params_to_file(wave4_filename, wave4)
        utilities.save_params_to_file(const3_filename, [params[4]])
        utilities.save_params_to_file(const4_filename, [params[5]])
        utilities.save_params_to_file(const5_filename, [params[6]])
        utilities.save_params_to_file(const6_filename, [params[7]])
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
            utilities.save_params_to_file(wave1_filename, wave1)
            utilities.save_params_to_file(wave2_filename, wave2)
            # utilities.save_params_to_file(wave3_filename, wave3)
            # utilities.save_params_to_file(wave4_filename, wave4)
            utilities.save_params_to_file(const3_filename, [params[4]])
            utilities.save_params_to_file(const4_filename, [params[5]])
            utilities.save_params_to_file(const5_filename, [params[6]])
            utilities.save_params_to_file(const6_filename, [params[7]])
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
                                       max_num_runs = 1000,
                                       target_cost = -1,
                                       num_params = 8,
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