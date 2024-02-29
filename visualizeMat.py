import pdb
from calculateSmallWorldnessParallel import calculate_small_worldness
from calculateSmallWorldnessParallel import generate_wrapped_diagonal_matrix
import concurrent.futures
from calculateSmallWorldnessParallel import calculate_small_worldness, generate_wrapped_diagonal_matrix
import matplotlib.pyplot as plt
import numpy as np

def read_diagonal_positions(filename):
    layer1PosLists = []
    layer2PosLists = []
    layer3PosLists = []

    with open(filename, 'r') as file:
        while True:
            # Attempt to read the next three lines (one block)
            block = [next(file, '').strip() for _ in range(3)]
            # If the first line of the block is empty, we've reached the end
            if not block[0]:
                break
            
            # Combine and clean the lines for the first matrix (removing brackets and splitting)
            layer1Lines = block[0]# + ' ' + block[1]
            
            layer1Pos = [int(pos.strip()) for pos in layer1Lines.replace('[', '').replace(']', '').split(',')]
            
            # Clean the lines for the second and third matrices and split
            layer2Pos = [int(pos.strip()) for pos in block[1].replace('[', '').replace(']', '').split(',')]
            layer3Pos = [int(pos.strip()) for pos in block[2].replace('[', '').replace(']', '').split(',')]
            
            # Append the positions to the respective lists
            layer1PosLists.append(layer1Pos)
            layer2PosLists.append(layer2Pos)
            layer3PosLists.append(layer3Pos)

    return layer1PosLists, layer2PosLists, layer3PosLists

def read_values_from_file(file_path):
    with open(file_path, 'r') as file:
        values = [float(line.strip()) for line in file.readlines()]
    return values

# Paths to your files
diagonals_file_path = '/p/dataset/abhishek/diag_pos_snnRun8a0.95noZero.txt'
accuracies_file_path = '/p/dataset/abhishek/max_accuracy_snnRun8a0.95noZero.txt'

# Reading from files
layer1PosList, layer2PosList, layer3PosList = read_diagonal_positions(diagonals_file_path)
maxAccList = read_values_from_file(accuracies_file_path)

#Get the index of max accuracy in maxAccList
maxAccIndex = maxAccList.index(max(maxAccList))
minAccIndex = maxAccList.index(min(maxAccList))

#Get the index of lowest n values in maxAccList and put them in a list
n = 12
lowestNIndices = np.argpartition(maxAccList, n)[:n]

#For the n indexes, use generate_wrapped_diagonal_matrix method to generate the matrices of dimension [300,784]
matrix1Min = [generate_wrapped_diagonal_matrix([300, 784], layer1PosList[lowestNIndices[i]]).toarray().astype(float) for i in range(len(lowestNIndices))]
pdb.set_trace()

#Make a figure using matrix1Min. Use subplots to plot all the matrices in one figure
fig, ax = plt.subplots(3, 4, figsize=(10, 6))
for i in range(len(lowestNIndices)):
    ax[i//4, i%4].imshow(matrix1Min[i], cmap='gray', interpolation='none')
    ax[i//4, i%4].set_title('Layer 1 Min Acc')
plt.savefig("layer1_min_snnRun8a.png")


'''matrix1Max = generate_wrapped_diagonal_matrix([300, 784], layer1PosList[maxAccIndex]).toarray().astype(float)
matrix1Min = generate_wrapped_diagonal_matrix([300, 784], layer1PosList[minAccIndex]).toarray().astype(float)
matrix2Max = generate_wrapped_diagonal_matrix([100, 300], layer2PosList[maxAccIndex]).toarray().astype(float)
matrix2Min = generate_wrapped_diagonal_matrix([100, 300], layer2PosList[minAccIndex]).toarray().astype(float)   

#Make a figure using matrix1Max and matrix1Min
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].imshow(matrix1Max, cmap='gray', interpolation='none')
ax[0].set_title('Layer 1 Max Acc (0.956)')
ax[1].imshow(matrix1Min, cmap='gray', interpolation='none')
ax[1].set_title('Layer 1 Min Acc (0.942)')
plt.savefig("layer1_max_min_snnRun8a.png")

#Do the same for matrix2
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].imshow(matrix2Max, cmap='gray', interpolation='none')
ax[0].set_title('Layer 2 Max Acc (0.956)')
ax[1].imshow(matrix2Min, cmap='gray', interpolation='none')
ax[1].set_title('Layer 2 Min Acc (0.942)')
plt.savefig("layer2_max_min_snnRun8aBand.png")'''
