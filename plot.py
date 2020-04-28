from matplotlib import   pyplot as plt
import os

# Choose which set to show
set_num = 5

PLOTS_DIR = os.path.join("plots")

# Choose if you want to see parameters
show_param = False 
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

if (set_num == 3):
    # Plot titles
    title = 'LM training using different word embeddings, lemmatised Yle corpus'
    xlabel = 'Epochs, test corpus ppl as the last data point'
    ylabel = 'ppl on lemmatised Yle validation corpus'
    plot_filename = 'lm_lemmatised.png'
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
    # x = list(range(7))
    # x = [num/7 for num in x]
    # print(x)
    x = [1,2,3,4,5,6,6.1]       # Iterations (epochs)
    

    # RESULTS
    # Y-axis
    set_size = 5
    label = [[]] * set_size
    results = [[]] * set_size
    # results[1] = [5456.77, 1148.65, 823.77, 701.62, 581.44, 526.70, 492.30]
    results[4] = [569.22, 475.07, 440.45, 418.07, 414.40, 415.90, 396.95] # Y results
    # results[0] = [6550.93, 1604.22, 1161.18, 1019.62, 897.73, 806.38, 755.81]
    results[0] = [841.72, 705.71, 653.69, 600.57, 561.23, 544.66, 534.87]
    results[2] = [594.75,486.83,454.01,437.73,432.07,426.36,408.09]
    results[3] = [593.73,487.31,445.89,432.99,423.78,420.88,403.03]
    results[1] = [602.97,501.78,472.51,445.28,442.46,446.55,428.52]

    label[4] = 'FastText, lemmatised IL'
    label[0] = 'Random initialisation'
    label[2] = 'Word2vec, lemmatised IL'
    label[3] = 'Word2vec, lemmatised IL and Wikipedia'
    label[1] = 'Word2vec, lemmatised Wikipedia'

if (set_num == 4):
    # Plot titles
    title = 'LM training using different word embeddings, non-lemmatised Yle corpus'
    xlabel = 'Epochs, test corpus ppl as the last data point' 
    ylabel = 'ppl on non-lemmatised Yle validation corpus'
    plot_filename = 'lm_nonlem.png'
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
    # x = list(range(7))
    # x = [num/7 for num in x]
    # print(x)
    x = [1,2,3,4,5,6,6.1]       # Iterations (epochs)
    

    # RESULTS
    # Y-axis
    set_size = 5
    label = [[]] * set_size
    results = [[]] * set_size

    results[0] = [2349.35,1871.78,1729.00,1561.93,1505.45,1432.48, 1400.80]
    results[1] = [1885.63,1505.36,1353.52,1284.19,1249.76,1241.69,1210.24]
    results[3] = [1457.00,1133.54,1044.64, 989.12,975.51,970.75,937.68]
    results[2] = [1563.02,1162.27,1045.88,1000.20,988.11, 990.97,959.56]
    results[4] = [1327.26,1050.57,949.60,905.01,880.74,883.35,857.69]
    # results[4] = 

    label[0] = 'Random initialisation'
    label[1] = 'Word2vec, lemmatised IL'
    label[3] = 'Word2vec, non-lemmatised IL'
    label[2] = 'FastText, lemmatised IL'
    label[4] = 'FastText, non-lemmatised IL'
    # label[3] = ''
    # label[4] = ''



if (set_num == 5):
    # Plot titles
    title = 'LM training using different word embeddings, raw text Yle corpus'
    xlabel = 'Epochs, test corpus ppl as last data point' 
    ylabel = 'ppl on raw text Yle validation corpus'
    plot_filename = 'lm_raw.png'
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
    # x = list(range(7))
    # x = [num/7 for num in x]
    # print(x)
    x = [1,2,3,4,5,6,6.1]       # Iterations (epochs)
    

    # RESULTS
    # Y-axis
    set_size = 3
    label = [[]] * set_size
    results = [[]] * set_size
    results[2] = [8357.24, 5941.33, 5511.79, 5107.41, 4976.71, 5017.11, 4905.87] # Y results
    results[0] = [18940.24, 15092.12, 11678.26, 10687.97, 10016.93, 9745.33, 9725.05]
    results[1] = [13174.51, 11044.94, 8832.67, 8352.72, 7986.14, 8088.95, 7915.84]
    # results[3] = 
    # results[4] = 

    label[2] = 'FastText, non-lemmatised IL'
    label[0] = 'Random initialisation'
    label[1] = 'Word2vec, lemmatised WP+IL'
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