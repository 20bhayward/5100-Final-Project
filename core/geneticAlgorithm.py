import random
import json
import matplotlib.pyplot as plt
import numpy as np

GOAL_DISTANCE = 500
JUMP_THRESHOLD = 30

def genetic_algorithm(agent, generations=20, population_size=10, num_actions=50, obstacles=None):
    population = initialize_population(population_size, num_actions)
    total_deaths = 0
    fitness_scores_per_generation = []
    goal_reaches_per_generation = []

    for generation in range(generations):
        fitness_scores = [evaluate_fitness(dna, agent, obstacles) for dna in population]
        total_deaths += sum(1 for score in fitness_scores if score < 0)
        fitness_scores_per_generation.append(max(fitness_scores))

        goal_reaches = sum(1 for score in fitness_scores if score >= 1000)  # Reaching the goal gives 1000 fitness
        goal_reaches_per_generation.append(goal_reaches)

        # Select the best agents for reproduction
        parents = select_parents(population, fitness_scores)

        # Generate new population through crossover and mutation
        population = reproduce(parents, population_size)

        if (generation + 1) % 100 == 0:
            print(f"Generation {generation + 1}: Best fitness = {max(fitness_scores)}")

    # Return the best DNA from the final population
    best_dna = population[fitness_scores.index(max(fitness_scores))]

    analyze_results(
    fitness_scores=fitness_scores_per_generation,
    total_evaluations=generations * population_size,
    generations=generations,
    deaths=total_deaths,
    population_size=population_size,
    goal_reaches_per_generation=goal_reaches_per_generation
)

    with open("best_dna.json", "w") as file:
        json.dump(best_dna, file)

    return best_dna, population

def initialize_population(size, num_actions):
    return [[random.choice(["jump", "wait"]) for _ in range(num_actions)] for _ in range(size)]

def get_nearest_trap_distance(agent, obstacles):
    nearest_distance = float('inf')
    for trap in obstacles:
        if trap.rect.x > agent.rect.right:  # Trap is ahead of the agent
            distance = trap.rect.x - agent.rect.right
            nearest_distance = min(nearest_distance, distance)
    return nearest_distance

def evaluate_fitness(dna, agent, obstacles):
    agent.reset()  # Reset the agent's state before running the DNA sequence
    fitness = 0
    JUMP_PENALTY_RANGE = 150

    for action in dna:
        nearby_traps = any(
            trap.rect.x > agent.rect.right and trap.rect.x < agent.rect.right + JUMP_PENALTY_RANGE
            for trap in obstacles
        )
        nearest_trap_distance = get_nearest_trap_distance(agent, obstacles)
        # print(nearest_trap_distance)
        if action == "jump":
            if nearest_trap_distance > JUMP_THRESHOLD:  # Penalize unnecessary jumps
                fitness -= 5
            agent.jump()  # Simulate a jump
        elif action == "wait" and nearest_trap_distance <= JUMP_THRESHOLD:
            fitness -= 10  # Penalize for waiting when a jump is needed

        for trap in obstacles:
            if agent.rect.colliderect(trap.rect):
                fitness -= 50  # Stop if the agent hits a trap
                return fitness
            if agent.rect.right > trap.rect.right:
                fitness += 20

        fitness += 1
        fitness += agent.rect.x * 0.1  # Use the x-coordinate as progress measurement
        
        if agent.rect.x >= GOAL_DISTANCE:
            fitness += 1000  # Large reward for reaching the goal
            print("Goal reached!")
            return fitness
        
        if agent.rect.y > 600:  # Assuming the screen height is 600
            fitness -= 50  # Penalize for falling
            print(f"Agent fell off at x={agent.rect.x}, fitness={fitness}")
            return fitness
        
    return fitness

def select_parents(population, fitness_scores):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
    elite = sorted_population[:2]  # Top 2 individuals
    rest = sorted_population[2:]
    probabilities = [score / sum(fitness_scores[2:]) for score in fitness_scores[2:]]
    parents = elite + random.choices(rest, probabilities, k=len(population) // 2 - 2)
    return parents

def reproduce(parents, population_size):
    offspring = []
    while len(offspring) < population_size:
        parent1, parent2 = random.sample(parents, 2)
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        offspring.append(mutate(child))
    return offspring

def mutate(dna, mutation_rate=0.3):
    for i in range(len(dna)):
        if random.random() < mutation_rate:
            dna[i] = random.choice(["jump", "wait"])
    return dna

def analyze_results(fitness_scores, total_evaluations, generations, deaths, population_size, goal_reaches_per_generation):
    """
    Analyze and visualize the results of genetic algorithm training.

    Args:
        fitness_scores (list): List of best fitness scores for each generation.
        total_evaluations (int): Total number of fitness evaluations.
        generations (int): Total number of generations.
        deaths (int): Number of deaths during training.
        population_size (int): Population size per generation.

    Returns:
        None
    """
    # Calculate success and failure rates
    success_rate = ((total_evaluations - deaths) / total_evaluations) * 100
    failure_rate = (deaths / total_evaluations) * 100
    overall_goal_reach_rate = (sum(goal_reaches_per_generation) / total_evaluations) * 100

    # Print metrics
    print(f"Total Evaluations: {total_evaluations}")
    print(f"Total Deaths: {deaths}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Failure Rate: {failure_rate:.2f}%")
    print(f"Goal Reach Rate: {overall_goal_reach_rate:.2f}%")

    # Plot fitness trend over generations
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, generations + 1), fitness_scores, marker="o", label="Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.title("Best Fitness Trend Over Generations")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot goal reach rate per generation
    plt.figure(figsize=(10, 6))
    goal_reach_rate_per_generation = [
        (goal_reaches / population_size) * 100 for goal_reaches in goal_reaches_per_generation
    ]
    plt.plot(range(1, generations + 1), goal_reach_rate_per_generation, marker="o", label="Goal Reach Rate")
    plt.xlabel("Generation")
    plt.ylabel("Goal Reach Rate (%)")
    plt.title("Goal Reach Rate Over Generations")
    plt.legend()
    plt.grid()
    plt.show()

    # Pie chart for success vs failure
    labels = ["Success", "Failure"]
    sizes = [success_rate, failure_rate]
    colors = ["#66b3ff", "#ff6666"]
    explode = (0.1, 0)  # "Explode" the success slice
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%", shadow=True, startangle=90)
    plt.title("Success vs Failure Rate")
    plt.show()