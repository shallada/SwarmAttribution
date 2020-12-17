import numpy as np
import random
import os.path
from os import path

POP_SIZE = 100
N_GEN = 900
N_PARENTS = 2
MUTATION_RATE = 0.02
TOURNAMENT_SIZE = 2


class GeneticLearner:

    def __init__(self, n_dim, feature_mask_size, state_file_name, use_baldwin_effect=False):
        self.n_dim = n_dim
        self.state_file_name = state_file_name
        self.starting_gen = 0
        self.use_baldwin_effect = use_baldwin_effect

        self.population = np.random.rand(POP_SIZE, n_dim)
        self.scores = np.zeros(POP_SIZE)
        self.feature_masks = np.zeros((POP_SIZE, feature_mask_size))

    def fit(self, x, y, target, n_gen = N_GEN):
        if not self.reclaim_state():
            self.starting_gen = 0
        self.evaluate_pop(x, y, target)
        self.sort_pop()
        for n in range(self.starting_gen, n_gen):
            self.breed(x, y, target)
            self.save_state(n)
            #if (n % 100) == 0:
            #    print(self.scores[0])
        return self.scores[0], self.population[0], self.feature_masks[0]

    def save_state(self, n):
        print("******************* SAVING STATE *********************")
        with open(self.state_file_name, 'wb') as f:
            np.save(f, [n])
            np.save(f, self.population)
            np.save(f, self.scores)

    def reclaim_state(self):
        exists = path.exists(self.state_file_name)
        if exists:
            with open(self.state_file_name, 'rb') as f:
                self.starting_gen = np.load(f)[0] + 1
                self.population = np.load(f)
                self.scores = np.load(f)
                print("******************* RECLAIMING STATE "+str(self.starting_gen)+" *********************")
        else:
            print("******************* NO STATE FOUND *********************")
        return exists

    def evaluate_pop(self, x, y, target):
        # Turn off the Baldin effect for initial population evaluation
        save_baldwin_state = self.use_baldwin_effect
        self.use_baldwin_effect = False

        for i in range(POP_SIZE):
            cnn_fitness, feature_mask = self.evaluate_individual(self.population[i], x, y, target)
            self.scores[i] = cnn_fitness
            self.feature_masks[i] = feature_mask

        self.use_baldwin_effect = save_baldwin_state

    def evaluate_individual(self, individual, x, y, target):
        target.set_weights(individual)
        if self.use_baldwin_effect:
            prev_mask = self.feature_masks[0] # if we are using Baldwin, then select the best mask
        else:
            prev_mask = None
        return target.score(x, y, prev_mask)

    def sort_pop(self):
        ind = np.flip(np.argsort(self.scores))
        self.population = self.population[ind]
        self.feature_masks[ind]
        self.scores = self.scores[ind]

    def breed(self, x, y, target):
        parents = self.choose_parents()
        child = self.breed_child(parents)
        cnn_fitness, feature_mask = self.evaluate_individual(child, x, y, target)
        if cnn_fitness > self.scores[-1]:
            self.population[-1] = child
            self.scores[-1] = cnn_fitness
            self.feature_masks[-1] = feature_mask
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
