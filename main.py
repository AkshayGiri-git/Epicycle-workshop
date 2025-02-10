from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import animation

import numpy as np
from numpy import interp

from math import tau
from scipy.integrate import quad
from scipy.spatial import distance_matrix

rc('animation', html='jshtml')
def create_close_loop(image_name, level=[200], binarize_threshold=128):
    # Prepare Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_aspect('equal', 'datalim')
    ax[1].set_aspect('equal', 'datalim')
    ax[0].set_title('Before Centered')
    ax[1].set_title('After Centered')

    # Step 1: Read image and convert to grayscale
    im = np.array(Image.open(image_name).convert('L'))

    # Step 2: Binarize the image using a threshold (default 128)
    binary_im = (im > binarize_threshold).astype(np.float64)

    # Step 3: Plot the original grayscale image and its contour
    contour_plot = ax[0].contour(binary_im, levels=level, colors='black', origin='image')

    # Step 4: Extract contour points (vertices) from the contour plot
    contour_path = contour_plot.get_paths()[0]
    x_table, y_table = contour_path.vertices[:, 0], contour_path.vertices[:, 1]

    # Step 5: Center the contour points around the origin
    x_table = x_table - min(x_table)
    y_table = y_table - min(y_table)
    x_table = x_table - max(x_table) / 2
    y_table = y_table - max(y_table) / 2

    # Step 6: Visualization of centered contour
    ax[1].plot(x_table, y_table, 'k-')

    # Step 7: Return time table and the transformed (centered) points
    time_table = np.linspace(0, 2 * np.pi, len(x_table))
    return time_table, x_table, y_table
def f(t, time_table, x_table, y_table):
    return interp(t, time_table, x_table) + 1j*interp(t, time_table, y_table)

def DFT(t, coef_list, order=5):
    kernel = np.array([np.exp(-n*1j*t) for n in range(-order, order+1)])
    series = np.sum( (coef_list[:,0]+1j*coef_list[:,1]) * kernel[:])
    return np.real(series), np.imag(series)

def coef_list(time_table, x_table, y_table, order=5):
    coef_list = []
    for n in range(-order, order+1):
        real_coef = quad(lambda t: np.real(f(t, time_table, x_table, y_table) * np.exp(-n*1j*t)), 0, tau, limit=500, full_output=1)[0]/tau
        imag_coef = quad(lambda t: np.imag(f(t, time_table, x_table, y_table) * np.exp(-n*1j*t)), 0, tau, limit=500, full_output=1)[0]/tau
        coef_list.append([real_coef, imag_coef])
    return np.array(coef_list)
def visualize(x_DFT, y_DFT, coef, order, space, fig_lim):
    fig, ax = plt.subplots()
    lim = max(fig_lim)
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_aspect('equal')

    # Initialize
    line = plt.plot([], [], 'k-', linewidth=2)[0]
    radius = [plt.plot([], [], 'r-', linewidth=0.5, marker='o', markersize=1)[0] for _ in range(2 * order + 1)]
    circles = [plt.plot([], [], 'r-', linewidth=0.5)[0] for _ in range(2 * order + 1)]

    def update_c(c, t):
        new_c = []
        for i, j in enumerate(range(-order, order + 1)):
            dtheta = -j * t
            ct, st = np.cos(dtheta), np.sin(dtheta)
            v = [ct * c[i][0] - st * c[i][1], st * c[i][0] + ct * c[i][1]]
            new_c.append(v)
        return np.array(new_c)

    def sort_velocity(order):
        idx = []
        for i in range(1,order+1):
            idx.extend([order+i, order-i])
        return idx

    def animate(i):
        # animate lines
        line.set_data(x_DFT[:i], y_DFT[:i])
        # animate circles
        r = [np.linalg.norm(coef[j]) for j in range(len(coef))]
        pos = coef[order]
        c = update_c(coef, i / len(space) * tau)
        idx = sort_velocity(order)
        for j, rad, circle in zip(idx,radius,circles):
            new_pos = pos + c[j]
            rad.set_data([pos[0], new_pos[0]], [pos[1], new_pos[1]])
            theta = np.linspace(0, tau, 50)
            x, y = r[j] * np.cos(theta) + pos[0], r[j] * np.sin(theta) + pos[1]
            circle.set_data(x, y)
            pos = new_pos

    # Animation
    ani = animation.FuncAnimation(fig, animate, frames=len(space), interval=15)
    return ani
time_table, x_table, y_table = create_close_loop("x.png",level=100)
x_table = 2 * (x_table - np.min(x_table)) / (np.max(x_table) - np.min(x_table)) - 1
y_table = 2 * (y_table - np.min(y_table)) / (np.max(y_table) - np.min(y_table)) - 1
order = 50
coef = coef_list(time_table, x_table, y_table, order)

space = np.linspace(0,tau,300)
x_DFT = [DFT(t,coef,order)[0] for t in space]
y_DFT = [DFT(t,coef, order)[1] for t in space]

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x_DFT, y_DFT, "r--")
ax.plot(x_table,y_table,"k-")
ax.set_aspect("equal", "datalim")
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

anim = visualize(x_DFT, y_DFT, coef, order, space, [xmin, xmax, ymin, ymax])
plt.show()
