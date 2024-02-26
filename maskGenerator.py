import numpy as np
import time
import math

def get_mask_pseudo_diagonal_numpy(mask_shape, sparsity, random_state=None,file_name=None):
  """Creates a random sparse mask with deterministic sparsity.

  Args:
    mask_shape: list, used to obtain shape of the random mask.
    sparsity: float, between 0 and 1.
    random_state: np.random.RandomState, if given the shuffle call is made using
      the RandomState

  Returns:
    numpy.ndarray
  """
  # Create an array of zeros with the specified shape
  mask = np.zeros(mask_shape)
  print("Sparsity is ", sparsity)
  if(sparsity != float(0)):
    elemBudget = (1 - sparsity)*mask_shape[0]*mask_shape[1]
  else:
    elemBudget = float(0)

  # Calculate the length of the diagonals
  #diag_length = min(mask_shape[0], mask_shape[1])
  diag_length = max(mask_shape[0], mask_shape[1])
  totalDiag = math.floor(float(elemBudget)/float(diag_length))
  if(mask_shape[0] == 10 and mask_shape[1] == 100):
      totalDiag = 1
      diag_length = mask_shape[1]
  print("Element budget is ", elemBudget)
  print("Total Diag count is ", totalDiag)
 
  print("Shape is ",mask_shape)

  #Set the main diagonal elements to ones
  #np.fill_diagonal(mask, 1)

  #TODO: Change it to depend on the sparsity
  #r = 6
  r = []

  np.random.seed(int(time.time()))

  # Determine custom sequence of starting positions
  start_positions = []
  used_rows = set()

  start_row = 0
  start_col = 0
  used_rows.add(0)

  start_positions.append((0,0))  
  
  for i in range(totalDiag-1):
    start_row = np.random.choice([row for row in range(mask_shape[0]) if row not in used_rows])
    used_rows.add(start_row)
    start_positions.append((start_row,start_col))
  
  r = [start_positions[i][0] for i in range(len(start_positions))]
  print(r)

  #pdb.set_trace()
  for start_row in r:
    current_row, current_col = (start_row)% mask_shape[0], 0
    #print(start_row, start_col)
    for _ in range(diag_length):
      mask[current_row % mask_shape[0], current_col % mask_shape[1]] = 1
      #print("mask",current_row % mask_shape[0], current_col % mask_shape[1])
      current_row += 1
      current_col += 1

      # Handle wrap-around logic for diagonals extending beyond the matrix width
      #if current_row == mask_shape[0]:
      #  current_col = mask_shape[1] - start_row

  with open('/p/dataset/abhishek/diag_pos_'+file_name+'.txt', 'a') as f:
    # Write the max accuracy to the file
    f.write(str(r))
    f.write("\n")

  print("Number of non-zeros: ", np.count_nonzero(mask))
  return mask

