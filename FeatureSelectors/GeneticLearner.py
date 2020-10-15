import numpy as np
import random


POP_SIZE = 100
N_GEN = 10000
N_PARENTS = 2
MUTATION_RATE = 0.02
TOURNAMENT_SIZE = 2

class GeneticLearner:

    def __init__(self, n_dim):
        self.n_dim = n_dim

        self.population = np.random.rand(POP_SIZE, n_dim)
        self.scores = np.zeros(POP_SIZE)

    def fit(self, x, y, target, n_gen = N_GEN):
        self.evaluate_pop(x, y, target)
        self.sort_pop()
        for n in range(n_gen):
            self.breed(x, y, target)
            #if (n % 100) == 0:
            #    print(self.scores[0])

    def evaluate_pop(self, x, y, target):
        for i in range(POP_SIZE):
            self.scores[i] = np.mean(self.evaluate_individual(self.population[i], x, y, target))

    def evaluate_individual(self, individual, x, y, target):
        target.set_weights(individual)
        return target.score(x, y)

    def sort_pop(self):
        ind = np.flip(np.argsort(self.scores))
        self.population = self.population[ind]
        self.scores = self.scores[ind]

    def breed(self, x, y, target):
        parents = self.choose_parents()
        child = self.breed_child(parents)
        accuracy = self.evaluate_individual(child, x, y, target)
        if accuracy > self.scores[-1]:
            self.population[-1] = child
            self.scores[-1] = accuracy
            self.sort_pop()

    def choose_parents(self):
        parents = []
        for _ in range(N_PARENTS):
            #parent = random.choice(self.population)
            max_score = -1
            for __ in range(TOURNAMENT_SIZE):
                i = random.randint(0, self.scores.size-1)
                if (self.scores[i] > max_score):
                    parent = self.population[i]
                    max_score = self.scores[i]
            parents.append(parent)
        return parents

    def breed_child(self, parents):
        child = np.empty(self.n_dim)
        for i in range(self.n_dim):
            if random.random() < MUTATION_RATE:
                child[i] = random.random()
            else:
                child[i] = random.choice(parents)[i]
        return child
