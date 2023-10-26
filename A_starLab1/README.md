# Set Covering  
## Problem Description
    N(ProblemSize) elements to cover  
    M(Numsets)  available sets
 
Each sets is created randomly , and it will cover a certain number of elements.
The goal is to cover all 'n' positions using the fewest possible sets.

## Problem Solution

The solution under consideration involves two distinct heuristics for the A* Algorithm.


The first heuristic utilizes a dictionary data structure for each state. In this structure, the key corresponds to the number of 'true elements,' and the value represents the available sets with that same number of true elements. It aims to estimate the minimum number of sets required to cover the remaining elements. This estimation begins with a key equivalent to the remaining elements and iterates downward to smaller values. This approach doesn't consider individual elements but rather estimates based on the total number of remaining elements."


The second heuristic employs a dictionary data structure for each state. In this structure, the keys represent the 'positions of elements,' and the corresponding values are tuples containing all the sets that cover those elements. The code estimates  a cost based on the size of the largest intersection that is not null,among the existing tuples of dictionary. 

 




 


Code developed jointly with [SilvioChito](https://github.com/SilvioChito/computational_intelligence) 