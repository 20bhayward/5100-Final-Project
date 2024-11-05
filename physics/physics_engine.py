# physics/physics_engine.py

class PhysicsEngine:
    """
    A class used to represent the Physics Engine.

    Methods
    -------
    apply_gravity(entity)
        Applies gravity to the given entity by calling its gravity method.

    handle_movement(entity, blocks)
        Handles the movement of the given entity by updating its position
        based on the provided blocks.
    """
    @staticmethod
    def apply_gravity(entity):
        """
        Applies gravity to the given entity.

        This function calls the gravity method on the provided entity object,
        which should update the entity's position or velocity based on the
        effects of gravity.

        Args:
            entity: An object that has a gravity method to apply gravitational effects.
        """
        entity.gravity()

    @staticmethod
    def handle_movement(entity, blocks):
        """
        Handles the movement of an entity by updating its position based on interactions with blocks.

        Args:
            entity: The entity whose movement is to be handled. It should have an update method.
            blocks: A list of blocks that the entity may interact with during movement.

        Returns:
            None
        """
        entity.update(blocks)
