import numpy as np
import scipy.sparse as sp
from calculateSmallWorldnessParallel import generate_wrapped_diagonal_matrix
from dataProcessing import read_diagonal_positions
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed


def calculate_jaccard_similarity(sparse_matrix_1, sparse_matrix_2):
    # Ensure the matrices are in COO format to easily access non-zero indices
    sm1_coo = sparse_matrix_1.tocoo()
    sm2_coo = sparse_matrix_2.tocoo()
    
    # Create sets of non-zero indices from each matrix
    non_zero_indices_1 = set(zip(sm1_coo.row, sm1_coo.col))
    non_zero_indices_2 = set(zip(sm2_coo.row, sm2_coo.col))
    
    # Calculate intersection and union
    intersection = non_zero_indices_1.intersection(non_zero_indices_2)
    union = non_zero_indices_1.union(non_zero_indices_2)
    
    # Compute Jaccard similarity coefficient
    jaccard_similarity = len(intersection) / len(union)
    
    return jaccard_similarity

# Example usage
#dim = (100, 100)  # Example dimensions
#density = 0.01  # Example density of non-zero elements

# Function to calculate jaccard similarity for a list of matrices
def calculate_jaccard_similarity_for_list(matrix_list):
    with ThreadPoolExecutor() as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(calculate_jaccard_similarity, matrix_list[0], matrix_list[i]) for i in range(1, len(matrix_list))]
        # Collect results as they complete
        results = [future.result() for future in as_completed(futures)]
    return results

# Paths to your files
diagonals_file_path = '/p/dataset/abhishek/diag_pos_snnRun8a0.95noZero.txt'
accuracies_file_path = '/p/dataset/abhishek/max_accuracy_snnRun8a0.95noZero.txt'

# Reading from files
layer1PosList, layer2PosList, layer3PosList = read_diagonal_positions(diagonals_file_path)

matrix1List = [generate_wrapped_diagonal_matrix([784,300], layer1PosList[i]) for i in range(len(layer1PosList))]
matrix2List = [generate_wrapped_diagonal_matrix([300,100], layer2PosList[i]) for i in range(len(layer2PosList))]
matrix3List = [generate_wrapped_diagonal_matrix([100,10], layer3PosList[i]) for i in range(len(layer3PosList))]

#Find the position of non-zero elements in the matrices in the lists matrix1List and matrix2List and matrix3List
nnzMatrix1List = [np.nonzero(matrix1List[i]) for i in range(len(matrix1List))]
nnzMatrix2List = [np.nonzero(matrix2List[i]) for i in range(len(matrix2List))]
nnzMatrix3List = [np.nonzero(matrix3List[i]) for i in range(len(matrix3List))]

# Use ThreadPoolExecutor to parallelize across lists
with ThreadPoolExecutor() as executor:
    # Submit tasks for each list
    future_to_matrix_list = {
        executor.submit(calculate_jaccard_similarity_for_list, matrix_list): matrix_list
        for matrix_list in [nnzMatrix1List, nnzMatrix2List, nnzMatrix3List]
    }
    # Wait for all tasks to complete and collect results
    for future in as_completed(future_to_matrix_list):
        matrix_list = future_to_matrix_list[future]
        try:
            result = future.result()
            # Now result contains the Jaccard similarities for the matrix_list
            # You can process the result here
        except Exception as exc:
            print(f'Generated an exception: {exc}')


'''#Calculate the jaccard similarity of first first entry in nnzMatrix1List with all other entries in the list one by one
jaccardSimilarityListL1 = [calculate_jaccard_similarity(matrix1List[0], matrix1List[i]) for i in range(1, len(matrix1List))]
jaccardSimilarityListL2 = [calculate_jaccard_similarity(matrix2List[0], matrix2List[i]) for i in range(1, len(matrix2List))]
jaccardSimilarityListL3 = [calculate_jaccard_similarity(matrix3List[0], matrix3List[i]) for i in range(1, len(matrix3List))]'''

#Write three plots, one for each layer, with jaccardsimilarities on the y axis and the matrix count on the x axis
plt.plot(jaccardSimilarityListL1)
plt.xlabel('Matrix Count')
plt.ylabel('Jaccard Similarity')
plt.title('Layer 1')
plt.savefig("jaccardSimilarityLayer1.png")

plt.plot(jaccardSimilarityListL2)
plt.xlabel('Matrix Count')
plt.ylabel('Jaccard Similarity')
plt.title('Layer 2')
plt.savefig("jaccardSimilarityLayer2.png")

plt.plot(jaccardSimilarityListL3)
plt.xlabel('Matrix Count')
plt.ylabel('Jaccard Similarity')
plt.title('Layer 3')
plt.savefig("jaccardSimilarityLayer3.png")