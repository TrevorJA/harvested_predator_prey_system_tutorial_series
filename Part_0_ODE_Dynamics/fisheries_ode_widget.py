"""
Author: Trevor Amestoy
Cornell University
Summer 2022

For waterprogramming.wordpress.com

This is used to generate an ipywidget which explores the dynamics of the
Lotka-Volterra system of equations for predator-prey populations.

Once the function is properly called, it will generate a widget with 'sliders'
for the variables; the user will be able to modify the value of the variables
using the sliders. The plots of ODE trajectories and populations will update
automatically.
"""

# Import core modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy

# Import the widget tools
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, HBox, Label, Layout, VBox, interactive_output

# Import functions from custom modules
from lotka_volterra_functions import type_II_lotka_volterra
from lotka_volterra_functions import type_II_equilibrium
from ode_plot_functions import plot_interactive_trajectories
from ode_plot_functions import plot_time_trajectories
from plot_inequality_surface import plot_inequality_surface

# Set style (unnecessary)
plt.style.use('ggplot')

# Ignore error: RuntimeWarning: invalid value encoutered in true_divide
np.seterr(divide = 'ignore', invalid = 'ignore')
scipy.special.seterr(all ='ignore')

def make_fisheries_widget():

    # Define some labels and style types for later use
    param_labs = ['a: prey availability', 'b: prey growth rate', 'c: rate that prey is converted to predator', 'd: predator death rate', 'h: handling time', 'K: prey carrying capacity', 'm: predator interference rate', 'z: harvesting effort']
    style = {'description_width': 'initial'}

    # Define sliders for later use: See Hadjimichael et al 2020 for the parameters and ranges
    a_slider = widgets.FloatSlider(value = 0.005, min = 0.002, max = 2.000, step = 0.001, disabled = False, continuous_update = False, orientation = 'horizontal', readout = True, readout_format = '0.2f', style = style)
    b_slider = widgets.FloatSlider(value = 0.500, min = 0.005, max = 1.000, step = 0.001, disabled = False, continuous_update = False, orientation = 'horizontal', readout = True, readout_format = '0.2f', style = style)
    c_slider = widgets.FloatSlider(value = 0.5, min = 0.2, max = 1.0, step = 0.1, disabled = False, continuous_update = False, orientation = 'horizontal', readout = True, readout_format = '0.2f', style = style)
    d_slider = widgets.FloatSlider(value = 0.10, min = 0.05, max = 0.20, step = 0.01, disabled = False, continuous_update = False, orientation = 'horizontal', readout = True, readout_format = '0.2f', style = style)
    h_slider = widgets.FloatSlider(value = 0.100, min = 0.001, max = 1.000, step = 0.001, disabled = False, continuous_update = False, orientation = 'horizontal', readout = True, readout_format = '0.2f', style = style)
    K_slider = widgets.FloatSlider(value = 2000, min = 100, max = 5000, step = 100,  disabled = False, continuous_update = False, orientation = 'horizontal', readout = True, readout_format = '0.2f', style = style)
    m_slider = widgets.FloatSlider(value = 0.7, min = 0.1, max = 1.5, step = 0.1,  disabled = False, continuous_update = False, orientation = 'horizontal', readout = True, readout_format = '0.2f', style = style)
    z_slider = widgets.FloatSlider(value = 0.00, min = 0.00, max = 1.00, step = 0.01,  disabled = False, continuous_update = False, orientation = 'horizontal', readout = True, readout_format = '0.2f', style = style)

    # Set up interactive plots using sliders
    trajectory_plot = interactive_output(plot_interactive_trajectories, {'a' : a_slider, 'b' : b_slider, 'c' : c_slider, 'd' : d_slider, 'h' : h_slider, 'K' : K_slider, 'm' : m_slider, 'z': z_slider})
    time_plot = interactive_output(plot_time_trajectories, {'a' : a_slider, 'b' : b_slider, 'c' : c_slider, 'd' : d_slider, 'h' : h_slider, 'K' : K_slider, 'm' : m_slider, 'z': z_slider})

    # Combine labels and plots into modular boxes
    box_layout = widgets.Layout(display='flex', flex_flow = 'row', align_items ='center', justify_content = 'center')
    both_plots = HBox([trajectory_plot, time_plot])
    top_row = HBox([VBox([Label(i) for i in param_labs]), VBox([a_slider, b_slider, c_slider, d_slider, h_slider, K_slider, m_slider, z_slider])], layout = box_layout)

    # Combine boxes into final widget
    fishery_widget = VBox([top_row, both_plots])

    return fishery_widget
