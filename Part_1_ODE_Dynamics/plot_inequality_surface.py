"""
Trevor Amestoy
Cornell University
Summer 2022

"""

import matplotlib.pyplot as plt
import numpy as np


def plot_inequality_surface(a, b, c, d, h, K, m, z):
    """
    Parameters:
    -----------

    Returns:
    --------
    """

    # Group parameters in a list
    params = [a, b, c, d, h, K, m, z]

    # Make a list of parameter labels
    param_labels = [str(lab) for lab in params]

    def check_inequality(a_val, b_val, h_val, K_val, m_val, z_val):
        "Returns the value of alpha (a) necessary to satisfy the inequality."
        if a_val < (((b_val - z_val)**m_val)/((h_val * K_val)**(1 - m_val))):
            stability = True
        else:
            stability = False
        return stability


    def inequality_surface(b_val, m_val):
        return (((b_val - z)**m_val)/(h*K)**(1-m_val))

    b_range = np.arange(0.005,1,0.001)
    m_range = np.arange(0.1,1.5,0.001)
    B, M = np.meshgrid(b_range, m_range)
    a_boundary = inequality_surface(B, M)
    A = a_boundary.clip(0,2)

    fig = plt.figure(figsize=(12,9))
    ax3D = fig.add_subplot(projection='3d')
    ax3D.plot_surface(B, M, A, color = 'grey', linewidth = 0, alpha = 0.1)

    # Choose the point color according to stability condition
    stability = check_inequality(a, b, h, K, m, z)
    if stability:
        point_color = 'green'
    else:
        point_color = 'red'


    ax3D.scatter(b, m, a, color = point_color, s = 80)


    ax3D.set_xlabel("b", fontsize = 40)
    ax3D.set_ylabel("m", fontsize = 40)
    ax3D.set_zlabel("a", fontsize = 40)
    ax3D.set_zlim([0.0,2.0])
    ax3D.set_xlim([0.0,1.0])
    ax3D.set_ylim([0.0,1.5])
    ax3D.xaxis.set_view_interval(0,  0.5)
    ax3D.set_facecolor('white')
    ax3D.view_init(12, -17)

    plt.show()
    return
