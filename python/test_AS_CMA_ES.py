import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from cma import CMAEvolutionStrategy
from AS_CMA_ES import AS_CMA_ES
from scipy.interpolate import interp1d

"""
test_AS_CMA_ES.py is a demonstration of AS-CMA-ES implemented in Python in a single
optimization run in a 2D. 
"""

# 0. Helper functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Steeper 2D bowl-shaped objective function
def objective_function(x1, x2): 
    f = x1**2 + x2**2+1
    return f

# Function to plot covariance ellipses
def plot_ellipse(ax, mean, cov, color):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, edgecolor=color, fc='None', lw=1)  # Thinner lines
    ax.add_patch(ellipse)

# Scaling and unscaling functions
def scale_to_bounds(x, lower_bound, upper_bound):
    scaled = lower_bound + x * (upper_bound - lower_bound)
    scaled = np.minimum(np.maximum(scaled, lower_bound), upper_bound)
    return scaled

def scale_to_01(x, lower_bound, upper_bound):
    return (x - lower_bound) / (upper_bound - lower_bound)

# 1. Initialize CMA-ES and landscape ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
N=2 # dimensionality
sigma = 0.3  # Initial step size
min_bound = -1
max_bound =  1
initial_mean = np.ones((N)) * 0.8  # Initial mean in 0-1 space
seed = 1
options = {'seed': seed}
np.random.seed(seed)
cmaes = CMAEvolutionStrategy(initial_mean, sigma, options)

# % 2. Initialize AS-CMA-ES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
use_adaptive_sampling = 1                   # boolean to use adaptive or static sampling
E_t = np.loadtxt('E(t).csv', delimiter=',')                  # read in error vs time file
if use_adaptive_sampling:
    beta = 1.3                              # desired signal-to-noise ratio, recommend 1.3
    y_hat = np.array([1, 500])              # estimated min/max possible cost that could be found in gen 1
    verboseType = 0                         # 0 turns off all AS-CMA prints
    as_cma = AS_CMA_ES(E_t, beta, y_hat, N, verboseType) # create as_cma object using above settings
else:
    static_sample_time = .5                 # if not using AS-CMA, this sets the time of all samples. 
                                            #   Generally, short static sample times converge to poor solutions quickly, 
                                            #   while long static sample times converge to quality solutions slowly

# % 3. Setup optimization loop ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
t_max = 250
t = 0 # time
g = 0 # generation
track_mean_cost = [objective_function(scale_to_bounds(cmaes.mean, min_bound, max_bound)[0], scale_to_bounds(cmaes.mean, min_bound, max_bound)[1])]
track_mean_param_value = [scale_to_bounds(cmaes.mean, min_bound, max_bound)]
track_sample_times = []
track_gen_times = [0]

candidates_all = []
covariances_all = []
means_all = []
arrows = []
prev_mean = scale_to_bounds(initial_mean, min_bound, max_bound)  # Scale initial mean to landscape bounds

# 4. Do optimization loop ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
while t < t_max:  # Run for specified generations
    candidates_01 = cmaes.ask()  # Ask for candidates in 0-1 space
    candidates_01[-1] = cmaes.mean
    candidates_01 = np.minimum(np.maximum(candidates_01, 0), 1)
    
    # Scale candidates from 0-1 to actual landscape bounds 
    candidates = [scale_to_bounds(cand, min_bound, max_bound) for cand in candidates_01]

    # get sampling time
    if use_adaptive_sampling:
        sample_times = as_cma.ask_all_sample_times(np.array(candidates_01).T)
    else:
        sample_times = np.ones((cmaes.popsize)) * static_sample_time
    
    # Store candidates and means
    candidates_all.append(candidates)
    means_all.append(scale_to_bounds(cmaes.mean, min_bound, max_bound))

    # Extract the covariance matrix (sigma^2 * C) and scale to landscape
    cov_scaled = cmaes.sigma**2 * cmaes.C
    cov = np.diag([8, 8]) @ cov_scaled @ np.diag([8, 8])  # Scale covariance to landscape bounds
    covariances_all.append(cov)
    
    # Sample the costs on the landscape with added noise
    costs_noisy = np.zeros((cmaes.popsize))
    for c in range(cmaes.popsize):
        cost_true = objective_function(candidates[c][0], candidates[c][1])
        noise_level = interp1d(E_t[0, :], E_t[1, :], kind='linear')(sample_times[c])
        noise_multiplier = np.maximum(1 + np.random.randn()*noise_level, 0)
        cost_noisy = cost_true * noise_multiplier
        costs_noisy[c] = cost_noisy
    
    # Store the true cost and mean value of the mean
    track_mean_cost.append(objective_function(scale_to_bounds(cmaes.mean, min_bound, max_bound)[0], scale_to_bounds(cmaes.mean, min_bound, max_bound)[1]))
    track_mean_param_value.append(scale_to_bounds(cmaes.mean, min_bound, max_bound))
    track_sample_times.append(sample_times)
    t += np.sum(sample_times)
    g += 1
    track_gen_times.append(t)

    # Return scaled candidates and costs to CMA-ES
    cmaes.tell(candidates_01, costs_noisy)
    if use_adaptive_sampling:
        as_cma.tell_generation_results(np.array(candidates_01).T, costs_noisy)
    
    # Store the arrow information for plotting later
    arrows.append((prev_mean, scale_to_bounds(cmaes.mean, min_bound, max_bound)))
    prev_mean = scale_to_bounds(cmaes.mean, min_bound, max_bound)

# 5. Evaluate results ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x = np.linspace(min_bound, max_bound, 1000)
y = np.linspace(min_bound, max_bound, 1000)
X, Y = np.meshgrid(x, y)
Z = objective_function(X, Y)

# Create plot with GridSpec
fig = plt.figure(figsize=(10, 6))  # Wider aspect ratio
gs = fig.add_gridspec(3, 6)  # 3 rows, 6 columns

# First subplot for optimization process (large and square)
ax1 = fig.add_subplot(gs[0:3, 0:4])  # Take the top three rows and first four columns
ax1.contourf(X, Y, Z, levels=5, cmap='Wistia')  # Lighter color map for better visibility

# Plot sampled points (smaller marker size)
for i in range(g):
    candidates = candidates_all[i]
    color =  [.9-.8*(i/g), .9-.8*(i/g), .9-.8*(i/g)] # Color for this generation
    size = 5
    ax1.scatter(*zip(*candidates), color=color, s=size) 

    # Plot covariance ellipse (thinner lines)
    plot_ellipse(ax1, means_all[i], covariances_all[i], color=color)

    ax1.scatter(means_all[i][0],means_all[i][1], color=color, marker='x')

    # Plot arrows showing movement from previous mean to current mean (black arrows)
    for prev_mean, current_mean in arrows[:i]:
        ax1.annotate('', xy=current_mean, xytext=prev_mean,
                    arrowprops=dict(facecolor='black', edgecolor='black', shrink=0.05, width=0.5, headwidth=5, linestyle='-', linewidth=1, alpha=0.8))

# Final plot settings: bound xlim and ylim to [-4, 4]
ax1.set_xlim([min_bound, max_bound])
ax1.set_ylim([min_bound, max_bound])
ax1.set_title('Cost landscape, means(x\'s) sampled points (dots), \n covariance matrices (ellipses)')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')

# Third subplot for mean cost over generations (smaller)
ax2 = fig.add_subplot(gs[1, 4:6])  # Take the second row and last two columns
ax2.plot(track_gen_times, track_mean_cost, marker='o', color='k',markersize=2)
ax2.set_title('True Cost of Mean')
ax2.set_yscale('log')
ax2.hlines(np.min(Z), 0, np.max(track_gen_times), linestyle='--', color='r')
ax2.grid()

# Second subplot for sample time over generations (smaller)
ax3 = fig.add_subplot(gs[0, 4:6])  # Take the first row and last two columns
ax3.scatter(np.cumsum(track_sample_times), np.array(track_sample_times).flatten(), marker='o', color='blue', s=1)
ax3.set_title('Sample time vs experiment time')
ax3.grid()
ax3.set_ylim([0.5, 5.5])

# Fourth subplot for mean values over generations (smaller)
ax4 = fig.add_subplot(gs[2, 4:6])  # Take the third row and last two columns
mean_x, mean_y = zip(*track_mean_param_value)
ax4.plot(track_gen_times, mean_x, marker='o', markersize=2, label='Mean x1', color='black')
ax4.plot(track_gen_times, mean_y, marker='o', markersize=2, label='Mean x2', color=[.5, .5, .5])
ax4.hlines(0, 0, np.max(track_gen_times), linestyle='--', color='r')
ax4.set_title('Mean Values')
ax4.set_xlabel('Generation')
ax4.set_ylabel('Mean Values')
ax4.legend()
ax4.grid()

plt.tight_layout()
if 1: 
    plt.show()
else:
    fig.savefig(f'./opt_result.png')

print('done')
    

