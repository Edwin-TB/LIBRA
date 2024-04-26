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
plt.rcParams['axes.grid'] = True

class Model():

    def __init__(self, initial_concentration, time_duration, source, mass_coeffs, areas, volume):
        """
        Args:
            initial_concentration (float): initial concnetration of tritium, typically 0. [] = count/s
            time_duration (float): How long the experiment lasts [] = seconds
            source (int): neutron source [] = count/s
            mass_coeffs (float list): list of mass transfer coefficients, currently in the following format:
                [k_top, k_wall] [] = m/s
            areas (float list): list of geometric areas, currently in the following format:
                [a_top, a_wall] [] = m^2
            volume (float): volume of reactor [] = m^3
        """
        self._initial_concentration = initial_concentration
        self._time_duration = time_duration
        self._source = source
        self._mass_coeffs = mass_coeffs
        self._areas = areas
        self._volume = volume

    ############ getter methods ############

    def get_initial_concentration(self):
        return self._initial_concentration
    def get_time_duration(self):
        return self._time_duration
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
        """
        Args: 
            current_time: point in time used to determine whether neutron source is on or not

        Returns:
            0 or 1: Integer multiplied by source, 0 for off and 1 for on
        """
        seconds_per_hour = 3600
        half_day = 12*seconds_per_hour
        full_day = 24*seconds_per_hour
        if 0 <= current_time <= half_day or full_day <= current_time <= full_day + half_day:
            return 1
        return 0
    
    def top_release(self): # GOOD
        k = self.get_mass_coeffs()
        A = self.get_areas()
        return k[0]*A[0]

    def wall_release(self): # GOOD
        k = self.get_mass_coeffs()
        A = self.get_areas()
        return k[1]*A[1]
        
    def model_function(self, current_concentration, current_time):

        S = self.source_func(current_time)*self.get_source()
        L = (self.top_release() + self.wall_release())
        c = current_concentration
        V = self.get_volume()
        dcdt = (S - L * c) / V
        return dcdt
    
class Data():
    def __init__(self, raw_data, timing, background, run_number):
        self._raw_data = raw_data
        self._timing = timing
        self._background = background
        self._run_number = run_number

    def get_raw_data(self):
        return self._raw_data
    def get_background(self):
        return self._background
    def get_timing(self):
        return self._timing
    def get_run_number(self):
        return self._run_number

    def remove_background(self):
        return self.get_raw_data() - self.get_background()
    
    def give_points(self):
        return np.sum(self.remove_background(), axis = 1)
    
def integrate_concentration(model):
    t_initial = 0
    t_final = (model.get_time_duration() + 1)
    t_step = 1
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


    initial_concentration = 0  
    days = 7
    hours_per_day = 24
    seconds_per_hour = 3600
    experiment_duration = days * hours_per_day * seconds_per_hour
    source = 3.65e5 
    mass_transfer_coeffs = [4.56e-7, 9.11e-8] # [top, wall]
    areas = [0.001380, 0.011618] # [top, wall]
    volume = 0.000100 

    model = Model(initial_concentration, experiment_duration, source, mass_transfer_coeffs, areas, volume)
    raw_data = np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]])
    timing = np.array([0, 1, 2, 3, 4, 5, 6, 7]) * hours_per_day * seconds_per_hour
    background = 0
    run_number = 7
    data_points = Data(raw_data, timing, background, run_number)
    
    concentration = integrate_concentration(model)[0]
    time = integrate_concentration(model)[1]
    top_release_values = model.top_release() * concentration
    wall_release_values = model.wall_release() * concentration

    cumulative_top_release = int.cumulative_trapezoid(np.asarray(top_release_values).squeeze(), time, initial=0)
    cumulative_wall_release = int.cumulative_trapezoid(np.asarray(wall_release_values).squeeze(), time, initial=0)

    activity = quantity_to_activity(concentration) * volume
    top_release_activity = quantity_to_activity(top_release_values)
    wall_release_activity = quantity_to_activity(wall_release_values)
    cumulative_top_release_activity = quantity_to_activity(cumulative_top_release)
    cumulative_wall_release_activity = quantity_to_activity(cumulative_wall_release)

    x_axis_points = np.arange(0,experiment_duration+1, hours_per_day * seconds_per_hour)
    x_axis_labels = np.arange(0,days+1,1)

    data_x = data_points.get_timing()
    data_y = data_points.give_points()

    fig, axs = plt.subplots(3, sharex = True, figsize = (10,10))
    axs[0].plot(time, activity)
    axs[0].set_title("Salt tritium inventory V*c_salt")
    axs[0].set_ylabel("Bq")
    axs[1].plot(time, top_release_activity)
    axs[1].plot(time, wall_release_activity)
    axs[1].set_title("Tritium Release Rate")
    axs[1].set_ylabel("Bq/s")
    axs[2].plot(time, cumulative_top_release_activity)
    axs[2].plot(time, cumulative_wall_release_activity)
    axs[2].plot(data_x, data_y,'ro')
    axs[2].set_xlabel("Days")
    axs[2].set_title("Cumulative Tritium Release")
    axs[2].set_ylabel("Bq")
    for ax in axs:
        ax.axvspan(0, 12*seconds_per_hour, alpha = 0.2, color = 'red')
        ax.axvspan(24*seconds_per_hour, 36*seconds_per_hour, alpha = 0.2, color = 'red')


    plt.xticks(x_axis_points,x_axis_labels)
    plt.show()


