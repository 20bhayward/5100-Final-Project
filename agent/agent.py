import random
import numpy as np
import pygame
from world.components.blocks.interactive.goal_block import GoalBlock

AGENT_COLOR = (0, 0, 255)

class Agent(pygame.sprite.Sprite):
    def __init__(self, x, y, genome_size=3, screen_height=600):
        super().__init__()
        self.width = 20
        self.height = 20
        self.image = pygame.Surface([self.width, self.height])
        self.image.fill(AGENT_COLOR)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        
        # Movement physics
        self.change_x = 0
        self.change_y = 0
        self.target_speed_x = 0
        self.acceleration = 0.5
        self.deceleration = 0.3
        self.max_speed_x = 6
        self.screen_height = screen_height
        
        # Jump physics
        self.jump_speed = -8.5
        self.gravity_acc = 0.35
        self.terminal_velocity = 12
        self.is_jumping = False
        self.jump_pressed = False
        
        # AI-related attributes
        self.genome = np.random.uniform(-1, 1, genome_size)
        self.fitness = 0
        self.mask = pygame.mask.from_surface(self.image)

    def update(self, blocks):
        # Apply horizontal acceleration/deceleration
        if self.target_speed_x != 0:
            # Accelerate towards target speed
            if self.change_x < self.target_speed_x:
                self.change_x = min(self.target_speed_x, self.change_x + self.acceleration)
            elif self.change_x > self.target_speed_x:
                self.change_x = max(self.target_speed_x, self.change_x - self.acceleration)
        else:
            # Decelerate to stop
            if abs(self.change_x) < self.deceleration:
                self.change_x = 0
            elif self.change_x > 0:
                self.change_x -= self.deceleration
            elif self.change_x < 0:
                self.change_x += self.deceleration

        # Apply horizontal movement
        self.rect.x += self.change_x

        # Horizontal collision detection
        block_hit_list = pygame.sprite.spritecollide(self, blocks, False, pygame.sprite.collide_mask)
        for block in block_hit_list:
            if isinstance(block, GoalBlock):
                continue
            
            if self.change_x > 0:
                self.rect.right = block.rect.left
                self.change_x = 0
            elif self.change_x < 0:
                self.rect.left = block.rect.right
                self.change_x = 0

        # Apply gravity and vertical movement
        self.gravity()
        self.rect.y += self.change_y

        # Vertical collision detection
        block_hit_list = pygame.sprite.spritecollide(self, blocks, False, pygame.sprite.collide_mask)
        for block in block_hit_list:
            if isinstance(block, GoalBlock):
                continue
            
            if self.change_y > 0:
                self.rect.bottom = block.rect.top
                self.change_y = 0
                self.is_jumping = False
            elif self.change_y < 0:
                self.rect.top = block.rect.bottom
                self.change_y = 0

    def gravity(self):
        if self.change_y == 0:
            self.change_y = 1
        else:
            self.change_y += self.gravity_acc
            # Limit falling speed
            if self.change_y > self.terminal_velocity:
                self.change_y = self.terminal_velocity

    def jump(self, blocks):
        # Check if on ground
        self.rect.y += 2
        block_hit_list = pygame.sprite.spritecollide(self, blocks, False)
        self.rect.y -= 2

        if (len(block_hit_list) > 0 or self.rect.bottom >= self.screen_height) and not self.is_jumping:
            self.change_y = self.jump_speed
            self.is_jumping = True

    def go_left(self):
        self.target_speed_x = -self.max_speed_x

    def go_right(self):
        self.target_speed_x = self.max_speed_x

    def stop(self):
        self.target_speed_x = 0

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

