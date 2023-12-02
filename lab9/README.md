# lab9.ipynb - Divide et Mutate

> In genetic algorithms a multi-part chromosome is a representation of an individual solution that is composed of multiple distinct segments or parts. Each part corresponds to a subset of the solution space, and genetic operators are applied independently to each.
> The primary motivation for using multi-part chromosomes is to exploit problem-specific structure or modularity. By dividing the solution into parts, the algorithm can focus on optimizing each part separately, potentially speeding up the convergence process. This approach is particularly useful when the problem exhibits some form of hierarchy, modularity, or separability.

Two fitness calls per iteration, one for the final.

## Progenitor Splitting - Segmentation

The `split_progenitor` function divides the progenitor into parts of length (`problem_instance`). This splitting process facilitates the independent evolution of different segments of the solution.

## Algorithm Execution

The `run` function executes the local search algorithm.

It involves the following steps:

- Creation of a random progenitor.
- Splitting the progenitor into parts.
- Iteratively mutating each part until its fitness reaches 1.0.
- Combining the evolved parts to form the final individual.
- Returning the number of fitness function calls.

## Why it is a Local Search Algorithm

The algorithm has characteristics that align with the principles of local search.

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

Shares similarities with Hill Climbing (basically HC for each subpart):

- _Local Exploration:_ Both algorithms focus on local exploration, making incremental changes to the current solution.
- _Memoryless:_ They do not maintain a memory of past solutions; decisions are based on the current state.
- _Exploitation:_ The algorithms exploit local improvements in fitness.

# lab9_bb - HC

HC algorithm with different acceptance rate for neighbors based on problem instance.

- Finds the best solution for the problem instance 1.
- Averages 0.5 for problem instance 2, 5, 10.

One fitness call per iteration.
