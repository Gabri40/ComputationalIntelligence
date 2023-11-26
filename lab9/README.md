# Divide et Mutate

## Components

### 1. Mutation Function

The `mutate` function is responsible for introducing changes to a given solution (`ind`) by randomly flipping the value of one gene. The mutation is accepted if it improves the fitness of the solution. This function serves as the primary mechanism for exploring the neighborhood of the current solution.

### 2. Progenitor Splitting

The `split_progenitor` function divides the progenitor, a randomly generated binary vector representing a potential solution, into parts of a specified length (`problem_instance`). This splitting process facilitates the independent evolution of different segments of the solution.

### 3. Algorithm Execution

The `run` function executes the local search algorithm. 

It involves the following steps:

- Creation of a random progenitor.
- Splitting the progenitor into parts.
- Iteratively mutating each part until its fitness reaches 1.0.
- Combining the evolved parts to form the final individual.
- Returning the number of fitness function calls and whether the individual meets a specified criterion.


## Why it is a Local Search Algorithm

The algorithm exhibits characteristics that align with the principles of local search.

1. **Iterative Improvement:**

   - The algorithm iteratively improves the current solution by making small changes to its genes.
   - The focus is on refining the solution locally rather than exploring the entire solution space.

2. **Neighborhood Exploration:**

   - The primary mechanism for exploring the solution space is the mutation function.
   - The mutation introduces small changes to a part of the solution, allowing the algorithm to explore nearby solutions.

3. **Exploitation of Local Optima:**
   - The algorithm is designed to exploit local optima by accepting mutations that lead to improved fitness.
   - It does not perform global exploration but rather exploits promising regions of the solution space.

### Relation to other Local Search Algorithms:

Shares similarities with Hill Climbing:

- _Local Exploration:_ Both algorithms focus on local exploration, making incremental changes to the current solution.
- _Memoryless:_ They do not maintain a memory of past solutions; decisions are based on the current state.
- _Exploitation:_ The algorithms exploit local improvements in fitness.


