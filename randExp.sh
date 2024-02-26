#!/bin/bash

# Loop 1000 times
for i in {1..3000}
do
   echo "Execution $i"
   ./training.sh
done

