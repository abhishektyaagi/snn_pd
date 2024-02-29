import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

'''# Function to read values from a file
def read_values_from_file(file_path):
    with open(file_path, 'r') as file:
        values = [float(line.strip()) for line in file.readlines()]
    return values

# Read SW values and accuracy values
sw_values = read_values_from_file('avgSW.txt')
accuracy_values = read_values_from_file('max_accuracy_run1.txt')

# Fit the values to a normal distribution
sw_mean, sw_std = norm.fit(sw_values)
accuracy_mean, accuracy_std = norm.fit(accuracy_values)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# SW Distribution
x_sw = np.linspace(min(sw_values), max(sw_values), 100)
p_sw = norm.pdf(x_sw, sw_mean, sw_std)
ax[0].plot(x_sw, p_sw, 'k', linewidth=2)
ax[0].hist(sw_values, density=True, alpha=0.6, color='b')
ax[0].set_title('SW Normal Distribution')
ax[0].set_xlabel('SW')
ax[0].set_ylabel('Density')

# Accuracy Distribution
x_accuracy = np.linspace(min(accuracy_values), max(accuracy_values), 100)
p_accuracy = norm.pdf(x_accuracy, accuracy_mean, accuracy_std)
ax[1].plot(x_accuracy, p_accuracy, 'k', linewidth=2)
ax[1].hist(accuracy_values, density=True, alpha=0.6, color='r')
ax[1].set_title('Accuracy Normal Distribution')
ax[1].set_xlabel('Accuracy')
ax[1].set_ylabel('Density')

plt.tight_layout()
plt.show()'''
import matplotlib.pyplot as plt

# Function to read values from a file
def read_values_from_file(file_path):
    with open(file_path, 'r') as file:
        values = [float(line.strip()) for line in file.readlines()]
    return values


# Read SW values and accuracy values
#sw_values = read_values_from_file('avgSW.txt')
swValLayer1 = read_values_from_file('./swValLayer1.txt')
swValLayer2 = read_values_from_file('./swValLayer2.txt')
swValLayer3 = read_values_from_file('./swValLayer3.txt')

accuracy_values = read_values_from_file('/p/dataset/abhishek/max_accuracy_snnRun8aBand.txt')

#If the len of accurac_values is not the same as the len of sw_values, then we need to remove the last element of the sw_values
if len(accuracy_values) != len(swValLayer1):
    swValLayer1 = swValLayer1

if len(accuracy_values) != len(swValLayer2):
    swValLayer2 = swValLayer2

if len(accuracy_values) != len(swValLayer3):
    swValLayer3 = swValLayer3
pdb.set_trace()
length = min(len(swValLayer1), len(swValLayer2), len(swValLayer3), len(accuracy_values))
# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(swValLayer1[:length], accuracy_values[:length], color='blue', alpha=0.5)
plt.title('Scatter Plot of Accuracy vs. Small-Worldness of Layer 1')
plt.xlabel('Small-Worldness (SW)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('accuracy_vs_sw_plot_layer1_snnRun8aBand.pdf')

plt.figure(figsize=(10, 6))
plt.scatter(swValLayer2[:length], accuracy_values[:length], color='blue', alpha=0.5)
plt.title('Scatter Plot of Accuracy vs. Small-Worldness of Layer 2')
plt.xlabel('Small-Worldness (SW)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('accuracy_vs_sw_plot_layer2_snnRun8aBand.pdf')

plt.figure(figsize=(10, 6))
plt.scatter(swValLayer3[:length], accuracy_values[:length], color='blue', alpha=0.5)
plt.title('Scatter Plot of Accuracy vs. Small-Worldness of Layer 3')
plt.xlabel('Small-Worldness (SW)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('accuracy_vs_sw_plot_layer3_snnRun8aBand.pdf')