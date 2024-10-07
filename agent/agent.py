import random
import numpy as np

class Agent:
    def __init__(self, genome_size):
        """
        Initialize the agent with a random genome.
        
        genome_size: The number of decision parameters (e.g., weights) in the genome.
        """
        # Genome represents the AI's decision-making process (like weights in a neural network)
        self.genome = np.random.uniform(-1, 1, genome_size)
        self.fitness = 0

    def get_action(self, inputs):
        """
        Use the agent's genome to decide an action based on its inputs.
        
        inputs: A list of sensory inputs (e.g., distance to obstacles).
        
        return: The action to be taken by the agent (e.g., move, jump).
        """
        # Weighted sum of inputs using the genome to make a decision
        weighted_sum = np.dot(inputs, self.genome)

        # Decide the action based on the weighted sum (threshold-based)
        if weighted_sum > 0.5:
            return "jump"  # Jump over an obstacle
        elif weighted_sum < -0.5:
            return "duck"  # Duck to avoid an obstacle
        else:
            return "run"  # Continue running

    def evaluate(self, survival_time, obstacles_dodged):
        """
        Evaluate the fitness of the agent based on its performance.
        
        survival_time: The time the agent survived without hitting obstacles.
        obstacles_dodged: Number of obstacles successfully avoided.
        
        return: The agent's fitness score.
        """
        # Fitness is based on how long the agent survived and how many obstacles it dodged
        self.fitness = survival_time + (obstacles_dodged * 2)
        return self.fitness

    def mutate(self, mutation_rate=0.1):
        """
        Apply mutation to the genome.
        
        mutation_rate: The probability of mutating a genome parameter.
        """
        for i in range(len(self.genome)):
            if random.random() < mutation_rate:
                # Mutate by adding a small random value to genome weights
                self.genome[i] += np.random.uniform(-0.5, 0.5)

    def crossover(self, other_agent):
        """
        Perform crossover with another agent to produce offspring.
        
        other_agent: The agent to cross genomes with.
        
        return: Two offspring agents.
        """
        crossover_point = random.randint(1, len(self.genome) - 1)

        # Perform crossover between two parents
        child1_genome = np.concatenate((self.genome[:crossover_point], other_agent.genome[crossover_point:]))
        child2_genome = np.concatenate((other_agent.genome[:crossover_point], self.genome[crossover_point:]))

        # Create offspring agents
        child1 = Agent(len(self.genome))
        child2 = Agent(len(self.genome))
        
        # Assign genomes to offspring
        child1.genome = child1_genome
        child2.genome = child2_genome
        
        return child1, child2
