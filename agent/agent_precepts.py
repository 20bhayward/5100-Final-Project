# agent_precepts.py
import pygame

class AgentPrecepts:
    def __init__(self, agent, level, pygame_manager):
        self.agent = agent
        self.level = level
        self.pygame_manager = pygame_manager

    def is_jump_necessary(self):
        """
        Determines if the agent needs to jump to avoid obstacles, traps, or gaps.
        """
        # Define a rectangle in front of the agent
        look_ahead_rect = pygame.Rect(
            self.agent.rect.right,
            self.agent.rect.bottom - self.agent.height,
            50,  # Look ahead distance
            self.agent.height
        )

        # Check for obstacles ahead
        obstacles_ahead = any(
            look_ahead_rect.colliderect(block.rect)
            for block in self.pygame_manager.block_list
        )

        # Check for traps ahead
        traps_ahead = any(
            look_ahead_rect.colliderect(trap.rect)
            for trap in self.level.trap_list
        )

        # Check for gaps (no ground below)
        ground_rect = pygame.Rect(
            self.agent.rect.x,
            self.agent.rect.bottom + 1,
            self.agent.width,
            5
        )
        ground_below = any(
            ground_rect.colliderect(block.rect)
            for block in self.pygame_manager.block_list
        )

        # Jump is necessary if there's an obstacle, trap ahead, or no ground below
        return obstacles_ahead or traps_ahead or not ground_below

    def get_goal_distance(self):
        """
        Calculate the Euclidean distance from the agent to the nearest goal.
        """
        goal = next(iter(self.level.goal_list))
        distance = ((goal.rect.centerx - self.agent.rect.centerx) ** 2 +
                    (goal.rect.centery - self.agent.rect.centery) ** 2) ** 0.5
        return distance

    def get_nearest_block_info(self):
        """
        Determines the distance to the nearest block in front of the agent and whether there is an obstacle in front.
        """
        look_ahead_rect = pygame.Rect(
            self.agent.rect.right,
            self.agent.rect.y,
            100,
            self.agent.height
        )
        obstacles = [block for block in self.pygame_manager.block_list if block.rect.colliderect(look_ahead_rect)]
        obstacle_in_front = 1.0 if obstacles else 0.0
        if obstacles:
            nearest_obstacle = min(obstacles, key=lambda block: block.rect.x)
            distance = nearest_obstacle.rect.x - self.agent.rect.x
        else:
            distance = 500.0  # Max look-ahead distance

        return distance, obstacle_in_front

    def is_trap_ahead(self):
        """
        Checks if there is a trap ahead of the agent.
        """
        # Define a rectangle in front of the agent
        look_ahead_rect = pygame.Rect(
            self.agent.rect.right,
            self.agent.rect.y,
            100,  # Look ahead distance
            self.agent.height
        )
        # Check for traps ahead
        traps_ahead = any(
            look_ahead_rect.colliderect(trap.rect)
            for trap in self.level.trap_list
        )
        return 1.0 if traps_ahead else 0.0

    def get_nearest_trap_distance(self):
        """
        Calculate the distance to the nearest trap in the agent's look-ahead path.
        """
        look_ahead_rect = pygame.Rect(
            self.agent.rect.right,
            self.agent.rect.y,
            500,  # Max look-ahead distance
            self.agent.height
        )
        traps = [trap for trap in self.level.trap_list if trap.rect.colliderect(look_ahead_rect)]
        if traps:
            nearest_trap = min(traps, key=lambda trap: trap.rect.x)
            distance = nearest_trap.rect.x - self.agent.rect.x
        else:
            distance = 500.0  # Max look-ahead distance
        return distance
