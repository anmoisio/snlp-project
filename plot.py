from matplotlib import   pyplot as plt
import os

set_num = 1                                  # Choose which set to show

PLOTS_DIR = os.path.join("plots")

show_param = True                            # Choose if you want to see parameters
plt.style.use('seaborn-whitegrid')
if (set_num == 1):
    title = 'Intrusion wrt. Iterations'
    xlabel = 'Iterations' 
    ylabel = 'Correct answers'
    plot_filename = 'intrusion_vs_iter.png'
    param_text_x = 6.8
    param_text_y = 440
    param_text_lineHeight = 2.2
    # Set 1 
    # Parameters
    parameter_size = 6
    parameters = [[]] * parameter_size
    parameters[0] = ['size: 200']
    parameters[1] = ['alpha: 0.025']
    parameters[2] = ['window: 5']
    parameters[3] = ['min_count: 2']
    parameters[4] = ['sg: 1']
    parameters[5] = ['negative: 5']

    # X-axis
    x = [7, 8, 9, 10, 12, 14, 16]        # Iterations (epochs)

    # RESULTS
    # Y-axis
    set_size = 5
    label = [[]] * set_size
    results = [[]] * set_size
    results[0] = [389, 397, 408, 400, 403, 409, 406] # Seed 1
    results[1] = [418, 437, 435, 438, 431, 440, 439] # Seed 2
    results[2] = [406, 410, 415, 419, 426, 423, 429] # Seed 3
    results[3] = [395, 395, 411, 411, 405, 410, 416] # Seed 4
    results[4] = [397, 395, 401, 413, 419, 428, 422] # Seed 5

    label[0] = 'Seed 1'
    label[1] = 'Seed 2'
    label[2] = 'Seed 3'
    label[3] = 'Seed 4'
    label[4] = 'Seed 5'
    
elif (set_num == 2):
    title = 'Analogy wrt. Iterations'
    plot_filename = 'analogy_vs_iter.png'
    xlabel = 'Iterations'
    ylabel = 'Correct answers'
    param_text_x = 6.8
    param_text_y = 375
    param_text_lineHeight = 2.5
    # Set 2
    # Parameters
    # Parameters
    parameter_size = 6
    parameters = [[]] * parameter_size
    parameters[0] = ['size: 200']
    parameters[1] = ['alpha: 0.025']
    parameters[2] = ['window: 5']
    parameters[3] = ['min_count: 2']
    parameters[4] = ['sg: 1']
    parameters[5] = ['negative: 5']

    # X-axis

    x = [7, 8, 9, 10, 12, 14, 16]    # Iterations (epochs)

    # RESULTS
    # Y-axis

    set_size = 1
    label = [[]] * set_size

    results = [[]] * set_size
    results[0] = [330, 337, 338, 319, 376, 343, 338] # Analogy 1

    label[0] = 'Analogy'

# Template
if (set_num == 3):
    # Plot titles
    title = 'Graph Title'
    xlabel = 'X-axis label' 
    ylabel = 'Y-axis label'
    plot_filename = 'filename.png'
    # Parameter label text positioning. These have to be set manually every time.
    param_text_x = 0                    # Parameter labels x position (wrt. the absolute values in x axis)
    param_text_y = 0                    # Parameter labels y position (wrt. the absolute values in y axis)
    param_text_lineHeight = 2.2         # Parameter labels line spacing
    # Set [#]
    # Parameter strings for parameter labels
    parameter_size = 6
    parameters = [[]] * parameter_size
    parameters[0] = ['size: 200']
    parameters[1] = ['alpha: 0.025']
    parameters[2] = ['window: 5']
    parameters[3] = ['min_count: 2']
    parameters[4] = ['sg: 1']
    parameters[5] = ['negative: 5']

    # X-axis
    x = []        # Iterations (epochs)

    # RESULTS
    # Y-axis
    set_size = 0
    label = [[]] * set_size
    results = [[]] * set_size
    results[0] = [] # Y results
    # results[1] = 
    # results[2] = 
    # results[3] = 
    # results[4] = 

    label[0] = ''
    # label[1] = ''
    # label[2] = ''
    # label[3] = ''
    # label[4] = ''

# Plotting
for i in range(0, set_size):   
    plt.plot(x, results[i], label = label[i])

# Labelling
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)

# Show
plt.grid(True)
plt.legend()
if (show_param == True):
    for i in range(0, parameter_size):
        plt.text(param_text_x, param_text_y, parameters[i][0])
        param_text_y -= param_text_lineHeight
plt.legend()

# Save the plot
plot_filename = os.path.join(PLOTS_DIR, plot_filename)
plt.savefig(plot_filename)

plt.show()