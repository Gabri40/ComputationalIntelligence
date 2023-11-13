# Evolutionary Strategy (ES) for evolving agents Nim Agents.

**1. Problem Description:**
The task involves creating agents to play Nim, a subtraction game where the goal is to avoid taking the last object. The game can have an arbitrary number of rows and a limit on the number of objects that can be removed in a turn.

**2. ES Algorithm Strategy:**

- **Parent Selection (μ):** The top 1/3 of the population is selected as parents for the subsequent generation.
- **Reproduction (ρ):** It generates one offspring (either by mutation or recombination) per selected parent. This corresponds to the "1" in the (μ/ρ, λ) notation.
- **Population Update:** The algorithm creates a new population by either mutating a randomly selected parent with a certain probability or generating an offspring through reproduction (mating) between randomly chosen parents.

**3. Components and Strategies:**

- The code includes various strategies for playing a Nim move, such as pure_random, gabriele, adaptive1, and optimal.
- The ES algorithm initializes a population of agents and evolves them over multiple generations. The fitness of an agent is determined by its performance in Nim matches against an expert agent that always does the optimal move.
- The algorithm combines mutation and reproduction to generate the next generation of agents. The mutation rate and population size are configurable parameters.

**4. Performance and Results:**

- The code provides periodic updates on the best fitness scores and average fitness across generations.
- The best agent, based on fitness, is identified and printed every 5 generations.
- The evolved agent's performance is evaluated by simulating 1000 matches against an expert agent.
