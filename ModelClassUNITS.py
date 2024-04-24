"""
This Python Library is my first attempt at consolidating LIBRA data analysis into classes and methods

It implements an Experiment base class that uses the basic numerical model:

V*dc/dt = S - L

In which S and L are the tritium source and loss respectively
"""

import numpy as np
import scipy.integrate as int
import matplotlib.pyplot as plt
from pint import UnitRegistry
ureg = UnitRegistry()

class Model():

    def __init__(self, initial_concentration, time_duration, time_step, source, mass_coeffs, areas, volume):
        """
        Args:
            time_i (float): initial time, typically 0
            time_f (float): final time
            source (function/method): neutron source and scheduling function
            mass_coeffs (float list): list of mass transfer coefficients, currently in the following format:
                [k_top, k_wall]
            areas (float list): list of geometric areas, currently in the following format:
                [a_top, a_wall]
            volume (float): volume of reactor
        
        """
        self._initial_concentration = initial_concentration
        self._time_duration = time_duration
        self._time_step = time_step
        self._source = source
        self._mass_coeffs = mass_coeffs
        self._areas = areas
        self._volume = volume

    ############ getter methods ############

    def get_initial_concentration(self):
        return self._initial_concentration
    def get_time_duration(self):
        return self._time_duration
    def get_time_step(self):
        return self._time_step
    def get_source(self):
        return self._source
    def get_mass_coeffs(self):
        return self._mass_coeffs
    def get_areas(self):
        return self._areas
    def get_volume(self):
        return self._volume
    
    ############ model-specific methods ############

    def source_func(self, current_time): # GOOD
        current_time = current_time * ureg.second
        if ((0 * ureg.hour) <= current_time <= (12 * ureg.hour)) or ((24 * ureg.hour) <= current_time <= (36 * ureg.hour)):
            return self.get_source()
        return 0
    
    def loss_func(self, current_concentration): 
        current_concentration = current_concentration * ureg.count / (ureg.meter ** 3)
        k = self.get_mass_coeffs()
        A = self.get_areas()
        print(current_concentration)
        return np.dot(k,A)*current_concentration
        
    def model_function(self, current_concentration, current_time):
        S = self.source_func(current_time)
        L = self.loss_func(current_concentration)
        V = self.get_volume()
        dcdt = (S-L)/V
        return dcdt
    
def integrate_concentration(model):
    t_initial = 0
    t_final = (model.get_time_duration() + model.get_time_step()).to(ureg.second).magnitude
    t_step = model.get_time_step().magnitude
    t = np.arange(t_initial, t_final, t_step)

    c0 = model.get_initial_concentration()

    tritium_concentration = int.odeint(model.model_function, c0, t)
    return tritium_concentration, t

def quantity_to_activity(quantity):
    avogadro = 6.022e23  # [] = mol^(-1)
    specific_activity = 3.57e14  # [] = Bq/g
    molar_mass = 3.016  # [] = g/mol
    return quantity * molar_mass * specific_activity / avogadro  # [] = Bq

if __name__ == '__main__':
    """
    From pset 1 of C20, the main method serves as a place to setup and 
    run anything from the 

    ex.
    model_1 = model_class(parameters)
    """
    initial_concentration = 0 * ureg.count / ureg.meter**3
    time_step = 1 * ureg.second
    experiment_duration = 1 * ureg.minute
    source = 3.65e5 * ureg.count / ureg.second
    mass_transfer_coeffs = [4.56e-7, 9.11e-8] * ureg.meter / ureg.second# [top, wall]
    areas = [13.80, 116.18]  * (ureg.centimeter ** 2) # [top, wall]
    volume = 100 * ureg.milliliter

    model = Model(initial_concentration, experiment_duration, time_step, source, mass_transfer_coeffs, areas, volume)
    concentration = integrate_concentration(model)[0]
    time = integrate_concentration(model)[1]

    fig, ax = plt.subplots()
    ax.plot(time, quantity_to_activity(concentration)*volume)
    ax.grid()
    plt.show()


