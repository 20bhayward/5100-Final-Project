# physics/physics_engine.py

class PhysicsEngine:
    @staticmethod
    def apply_gravity(entity):
        entity.gravity()

    @staticmethod
    def handle_movement(entity, blocks):
        entity.update(blocks)
