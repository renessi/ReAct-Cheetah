class FactMemory:

    def __init__(self):
        self.entities = {}

    def store(self, entity: str, data: dict):

        if entity not in self.entities:
            self.entities[entity] = {}

        self.entities[entity].update(data)

    def get(self, entity: str):

        return self.entities.get(entity)

    def has(self, entity: str):

        return entity in self.entities