import pdb
from calculateSmallWorldnessParallel import calculate_small_worldness
from calculateSmallWorldnessParallel import generate_wrapped_diagonal_matrix
import concurrent.futures
from calculateSmallWorldnessParallel import calculate_small_worldness, generate_wrapped_diagonal_matrix

def calculate_small_worldness_wrapper(matrix):
    return calculate_small_worldness(matrix)

def parallel_calculate_small_worldness(matrix_list):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(calculate_small_worldness_wrapper, matrix_list))
    return results

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

def read_accuracies(file_path):
    max_acc_list = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                accuracy = float(line.strip())
                if 0 <= accuracy <= 1:
                    max_acc_list.append(accuracy)
            except ValueError:
                pass  # Ignore lines that cannot be converted to float
    return max_acc_list

# Paths to your files
diagonals_file_path = '/p/dataset/abhishek/diag_pos_snnRun8aBand.txt'
accuracies_file_path = '/p/dataset/abhishek/max_accuracy_snnRun8aBand.txt'

# Reading from files
layer1PosList, layer2PosList, layer3PosList = read_diagonal_positions(diagonals_file_path)

maxAccList = read_accuracies(accuracies_file_path)

# Example output
'''print("Layer 1 Positions:", layer1PosList)
print("Layer 2 Positions:", layer2PosList)
print("Layer 3 Positions:", layer3PosList)
print("Max Accuracies:", maxAccList)'''

# Calculating small-worldness for each layer
matrix1List = [generate_wrapped_diagonal_matrix([784,300], layer1PosList[i]) for i in range(len(layer1PosList))]
matrix2List = [generate_wrapped_diagonal_matrix([300,100], layer2PosList[i]) for i in range(len(layer2PosList))]
matrix3List = [generate_wrapped_diagonal_matrix([100,10], layer3PosList[i]) for i in range(len(layer3PosList))]

#swValLayer1 = [calculate_small_worldness(matrix1List[i]) for i in range(len(matrix1List))]
#swVallayer2 = [calculate_small_worldness(matrix2List[i]) for i in range(len(matrix2List))]
#swValLayer3 = [calculate_small_worldness(matrix3List[i]) for i in range(len(matrix3List))]

'''resultsL1 = [calculate_small_worldness(matrix1List[i]) for i in range(len(matrix1List))]
resultsL2 = [calculate_small_worldness(matrix2List[i]) for i in range(len(matrix2List))]
resultsL3 = [calculate_small_worldness(matrix3List[i]) for i in range(len(matrix3List))]'''

# Parallel calculation of small-worldness for each layer
resultsL1 = parallel_calculate_small_worldness(matrix1List)
resultsL2 = parallel_calculate_small_worldness(matrix2List)
resultsL3 = parallel_calculate_small_worldness(matrix3List)

C1,L1 = zip(*resultsL1)
C2,L2 = zip(*resultsL2)
C3,L3 = zip(*resultsL3)

swValLayer1 = [(C1[i]/0.5)/(L1[i]/10) for i in range(len(C1))]
swValLayer2 = [(C2[i]/0.5)/(L2[i]/10) for i in range(len(C2))]
swValLayer3 = [(C3[i]/0.5)/(L3[i]/10) for i in range(len(C3))]

fileSW1 = './swValLayer1.txt'
fileSW2 = './swValLayer2.txt'   
fileSW3 = './swValLayer3.txt'

fileC1 = './C1.txt'
fileL1 = './L1.txt'
fileC2 = './C2.txt'
fileL2 = './L2.txt'
fileC3 = './C3.txt'
fileL3 = './L3.txt'

with open(fileC1, 'w') as file:
    for value in C1:
        value_to_write = str(value) + '\n'
        file.write(value_to_write)

with open(fileL1, 'w') as file:
    for value in L1:
        value_to_write = str(value) + '\n'
        file.write(value_to_write)

with open(fileC2, 'w') as file: 
    for value in C2:
        value_to_write = str(value) + '\n'
        file.write(value_to_write)

with open(fileL2, 'w') as file: 
    for value in L2:
        value_to_write = str(value) + '\n'
        file.write(value_to_write)

with open(fileC3, 'w') as file:
    for value in C3:
        value_to_write = str(value) + '\n'
        file.write(value_to_write)  

with open(fileL3, 'w') as file: 
    for value in L3:
        value_to_write = str(value) + '\n'
        file.write(value_to_write)

with open(fileSW1, 'w') as file:
    for value in swValLayer1:
        value_to_write = str(value) + '\n'
        file.write(value_to_write)

with open(fileSW2, 'w') as file:
    for value in swValLayer2:
        value_to_write = str(value) + '\n'
        file.write(value_to_write)
    
with open(fileSW3, 'w') as file:
    for value in swValLayer3:
        value_to_write = str(value) + '\n'
        file.write(value_to_write)

'''avgSW = [(swValLayer1[i] + swVallayer2[i] + swValLayer3[i])/3 for i in range(len(swValLayer1))]
file_path = './avgSW.txt'

with open(file_path, 'w') as file:
    for value in avgSW:
        value_to_write = str(value) + '\n'
        file.write(value_to_write)

'''
#print("Average Small-Worldness:", avgSW)