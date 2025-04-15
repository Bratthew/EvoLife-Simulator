import numpy as np
import torch
import torch.nn as nn
import deap.base
import deap.creator
import deap.tools
import random
import matplotlib.pyplot as plt
from copy import deepcopy
np.random.seed(42)
random.seed(42)

GRID_SIZE = 20
FOOD_COUNT = 50
MAX_STEPS = 100
POP_SIZE = 10
HATCH_TIME = 10

class SeekBrain(nn.Module):
    def __init__(self):
        super(SeekBrain, self).__init__()
        self.fc1 = nn.Linear(6, 10)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Seek:
    def __init__(self, x, y, brain=None):
        self.x = x
        self.y = y
        self.speed = 0.0 if brain is None else random.uniform(0, 2)
        self.size = 1.0 if brain is None else random.uniform(0.5, 2)
        self.color = (random.random(), random.random(), random.random())
        self.metabolism = 0.5 if brain is None else random.uniform(0.1, 1)
        self.energy = 100
        self.health = 100
        self.age = 0
        self.brain = SeekBrain() if brain is None else brain
        self.alive = True

    def get_inputs(self, grid):
        inputs = [self.x / GRID_SIZE, self.y / GRID_SIZE]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] == 1:
                inputs.append(1)
            else:
                inputs.append(0)
        return torch.tensor(inputs, dtype=torch.float32)
    
    def move(self, grid):
        if not self.alive:
            return
        inputs = self.get_inputs(grid)
        outputs = self.brain(inputs)
        action = torch.argmax(outputs).item()
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)][action]
        new_x = self.x + int(dx * self.speed)
        new_y = self.y + int(dy * self.speed)
        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
            self.x, self.y = new_x, new_y
        self.energy -= self.metabolism * self.size
        self.age += 1
        if self.energy <= 0:
            self.health -= 10
        if self.health <= 0 or self.age >= 100:
            self.alive = False

    def eat(self, grid):
        if grid[self.x, self.y] == 1:
            self.energy += 20
            grid[self.x, self.y] = 0
            return True
        return False
    
    def lay_egg(self):
        if not self.alive or self.age < 20:
            return None
        p_egg = (self.energy / 150) * (self.health / 100) * 0.05
        if random.random() < p_egg:
            egg = Seek(self.x, self.y, brain=deepcopy(self.brain))
            egg.speed = max(0, self.speed + random.gauss(0, 0.1))
            egg.size = max(0.5, min(2, self.size + random.gauss(0, 0.1)))
            egg.metabolism = max(0.1, min(1, self.metabolism + random.gauss(0, 0.05)))
            egg.color = (
                max(0, min(1, self.color[0] + random.gauss(0, 0.1))),
                max(0, min(1, self.color[1] + random.gauss(0, 0.1))),
                max(0, min(1, self.color[2] + random.gauss(0, 0.1)))
            )

            for param in egg.brain.parameters():
                param.data += torch.randn_like(param) * 0.1
            return egg
        return None
    
class Environment:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.seeks = []
        self.eggs = []
        self.food_eaten = 0
        self.place_food()

    def place_food(self):
        for _ in range(FOOD_COUNT):
            x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
            self.grid[x, y] = 1

    def step(self):
        for seek in self.seeks[:]:
            if seek.alive:
                seek.move(self.grid)
                if seek.eat(self.grid):
                    self.food_eaten += 1
                egg = seek.lay_egg()
                if egg:
                    self.eggs.append((egg, 0))
            else:
                self.seeks.remove(seek)

        for egg, age in self.eggs[:]:
            if age >= HATCH_TIME:
                self.seeks.append(egg)
                self.eggs.remove((egg, age))
            else:
                self.eggs[self.eggs.index((egg, age))] = (egg, age + 1)
        if self.food_eaten > 0:
            self.place_food()
            self.food_eaten -= 1

    def add_seek(self, seek):
        self.seeks.append(seek)

deap.creator.create("FitnessMax", deap.base.Fitness, weights=(1.0,))
deap.creator.create("Individual", list, fitness=deap.creator.FitnessMax)

def evaluate_individual(individual):
    brain = SeekBrain()
    idx = 0
    for param in brain.parameters():
        param_shape = param.data.shape
        param_size = param.data.numel()
        param.data = torch.tensor(individual[idx:idx + param_size], dtype=torch.float32).reshape(param_shape)
        idx += param_size
    seek = Seek(random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1), brain=brain)

    env = Environment()
    env.add_seek(seek)
    fitness = 0
    for _ in range(MAX_STEPS):
        env.step()
        if not seek.alive:
            break
        fitness += seek.energy / 100 + seek.age / 100
    return fitness,

toolbox = deap.base.Toolbox()
brain = SeekBrain()
total_weights = sum(p.numel() for p in brain.parameters())
toolbox.register("attr_float", random.gauss, 0, 1)
toolbox.register("individual", deap.tools.initRepeat, deap.creator.Individual, toolbox.attr_float, n=total_weights)
toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", deap.tools.cxBlend, alpha=0.5)
toolbox.register("mutate", deap.tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", deap.tools.selTournament, tournsize=3)

def run_simulation():
    pop = toolbox.population(n=POP_SIZE)
    stats = deap.tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.values else 0)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    env = Environment()
    for ind in pop:
        brain = SeekBrain()
        idx = 0
        for param in brain.parameters():
            param_shape = param.data.shape
            param_size = param.data.numel()
            param.data = torch.tensor(ind[idx:idx+param_size], dtype=torch.float32).reshape(param_shape)
            idx += param_size
        seek = Seek(random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1), brain=brain)
        env.add_seek(seek)

    plt.ion()
    fig, ax = plt.subplots()
    for gen in range(50):
        for step in range(MAX_STEPS):
            env.step()
            if step % 10 == 0:
                ax.clear()
                food_x, food_y, = np.where(env.grid == 1)
                ax.scatter(food_y, food_x, c='green', s=10, label='Food')
                for seek in env.seeks:
                    ax.scatter(seek.y, seek.x, c=[seek.color], s=seek.size*50, alpha=0.6)
                for egg, _ in env.eggs:
                    ax.scatter(egg.y, egg.x, c='gray', s=egg.size*30, marker='^')
                ax.set_xlim(-1, GRID_SIZE)
                ax.set_ylim(-1, GRID_SIZE)
                ax.set_title(f"Generation {gen+1}, Step {step+1}")
                plt.pause(0.01)

        fitnesses = []
        for ind_idx, ind in enumerate(pop):
            seek = env.seeks[ind_idx] if ind_idx < len(env.seeks) else None
            if seek and seek.alive:
                egg_count = sum(1 for egg, _ in env.eggs if egg.x == seek.x and egg.y == seek.y)
                fitness = seek.age + seek.energy / 100 + egg_count
            else:
                fitness = 0.1 * random.random()
            fitnesses.append((fitness,))
        
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        record = stats.compile(pop)
        print(f"Gen {gen+1}: {record}, Seeks: {len(env.seeks)}, Eggs: {len(env.eggs)}")

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        pop[:] = offspring

        env = Environment()
        for ind in pop:
            brain = SeekBrain()
            idx = 0
            for param in brain.parameters():
                param_shape = param.data.shape
                param_size = param.data.numel()
                param.data = torch.tensor(ind[idx:idx+param_size], dtype=torch.float32).reshape(param_shape)
                idx += param_size
            seek = Seek(random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1), brain=brain)
            env.add_seek(seek)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_simulation()
