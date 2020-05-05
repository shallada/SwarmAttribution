import numpy as np
import random
import sys
from copy import deepcopy
from scipy import optimize


from EvaluateMask import EvaluateMask
from LoadFeatures import LoadFeatures

if len(sys.argv) != 2:
	print("include the dataset subfolder name as a parameter")
	exit()

#
# Parameters
#
all_features_file_name = "../Concatenator/"+sys.argv[1]+"/AllFeatures.txt"
ColonySize = 10
NIters = 50
MaxTrial = 10
UseDiscrete = True
MaskMin = 0.0
MaskMax = 1.0
ChangeProb = 0.02 # probability of changing a bit when exploring
NRuns = 30
FeatureWeight = 0.0


class ObjectiveFunction(object):

	def __init__(self, name, dim, minf, maxf):
		self.name = name
		self.dim = dim
		self.minf = minf
		self.maxf = maxf

	def sample(self):
		return np.random.uniform(low=self.minf, high=self.maxf, size=self.dim)

	def custom_sample(self):
		if UseDiscrete:
			mask = np.random.choice([0, 1], size=(self.dim,)).tolist()
		else:
			mask = [random.random() for j in range(self.dim)]

		"""
		return np.repeat(self.minf, repeats=self.dim) \
			   + np.random.uniform(low=0, high=1, size=self.dim) *\
			   np.repeat(self.maxf - self.minf, repeats=self.dim)
		"""
		return mask


	def evaluate(self, x):
		pass


class MaskEvaluator(ObjectiveFunction):

	def __init__(self, dim, x, y):
		super(MaskEvaluator, self).__init__('MaskEvaluator', dim, 0.0, 1.0)
		self.x = x
		self.y = y

	def evaluate(self, mask):
		return EvaluateMask(mask, self.x, self.y, feature_weight = FeatureWeight)

class ArtificialBee(object):

	TRIAL_INITIAL_DEFAULT_VALUE = 0
	INITIAL_DEFAULT_PROBABILITY = 0.0

	def __init__(self, obj_function):
		self.pos = obj_function.custom_sample()
		self.obj_function = obj_function
		self.minf = obj_function.minf
		self.maxf = obj_function.maxf
		self.fitness = obj_function.evaluate(self.pos)
		self.trial = ArtificialBee.TRIAL_INITIAL_DEFAULT_VALUE
		self.prob = ArtificialBee.INITIAL_DEFAULT_PROBABILITY

	def evaluate_boundaries(self, pos):
		if (pos < self.minf).any() or (pos > self.maxf).any():
			pos[pos > self.maxf] = self.maxf
			pos[pos < self.minf] = self.minf
		return pos

	def update_bee(self, pos, fitness):
		if fitness > self.fitness:
			self.pos = pos
			self.fitness = fitness
			self.trial = 0
		else:
			self.trial += 1

	def reset_bee(self, max_trials):
		if self.trial >= max_trials:
			self.__reset_bee()

	def __reset_bee(self):
		self.pos = self.obj_function.custom_sample()
		self.fitness = self.obj_function.evaluate(self.pos)
		self.trial = ArtificialBee.TRIAL_INITIAL_DEFAULT_VALUE
		self.prob = ArtificialBee.INITIAL_DEFAULT_PROBABILITY


class EmployeeBee(ArtificialBee):

	def explore(self, max_trials):
		#print("self trial = "+str(self.trial)+", max trial = "+str(max_trials))
		if self.trial <= max_trials:
			if UseDiscrete:
				n_pos = self.pos.copy()
				for i in range(len(self.pos)):
					if np.random.random() < ChangeProb:
						n_pos[i] = np.random.choice([0, 1])
			else:
				component = np.random.choice(self.pos)
				phi = np.random.uniform(low=-1, high=1, size=len(self.pos))
				n_pos = self.pos + (self.pos - component) * phi
				n_pos = self.evaluate_boundaries(n_pos)
			n_fitness = self.obj_function.evaluate(n_pos)
			#print("############ n_fitness = "+str(n_fitness))
			self.update_bee(n_pos, n_fitness)

	def get_fitness(self):
		return 1 / (1 + self.fitness) if self.fitness >= 0 else 1 + np.abs(self.fitness)

	def compute_prob(self, max_fitness):
		self.prob = self.get_fitness() / max_fitness

class OnLookerBee(ArtificialBee):

	def onlook(self, best_food_sources, max_trials):
		candidate = np.random.choice(best_food_sources)
		self.__exploit(candidate.pos, candidate.fitness, max_trials)

	def __exploit(self, candidate, fitness, max_trials):
		if self.trial <= max_trials:
			if UseDiscrete:
				n_pos = self.pos.copy()
				for i in range(len(self.pos)):
					if np.random.random() < ChangeProb:
						n_pos[i] = np.random.choice([0, 1])
			else:
				component = np.random.choice(self.pos)
				phi = np.random.uniform(low=-1, high=1, size=len(self.pos))
				n_pos = self.pos + (self.pos - component) * phi
				n_pos = self.evaluate_boundaries(n_pos)
			n_fitness = self.obj_function.evaluate(n_pos)
			self.update_bee(n_pos, n_fitness)


			if n_fitness <= fitness:
				self.pos = n_pos
				self.fitness = n_fitness
				self.trial = 0
			else:
				self.trial += 1



class ArtificialBeeColony(object):

	def __init__(self, obj_function, colony_size, n_iter, max_trials):
		self.colony_size = colony_size
		self.obj_function = obj_function
		self.n_iter = n_iter
		self.max_trials = max_trials

		self.optimal_solution = None
		self.optimality_tracking = []

	def __reset_algorithm(self):
		self.optimal_solution = None
		self.optimality_tracking = []

	def __update_optimality_tracking(self):
		self.optimality_tracking.append(self.optimal_solution.fitness)

	def __update_optimal_solution(self):
		n_optimal_solution = max(self.onlokeer_bees + self.employee_bees, key=lambda bee: bee.fitness)
		if not self.optimal_solution:
			self.optimal_solution = deepcopy(n_optimal_solution)
		else:
			#print("n_opt = "+str(n_optimal_solution.fitness)+", opt = "+str(self.optimal_solution.fitness))
			if n_optimal_solution.fitness > self.optimal_solution.fitness:
				self.optimal_solution = deepcopy(n_optimal_solution)

	def __initialize_employees(self):
		self.employee_bees = []
		for itr in range(self.colony_size // 2):
			self.employee_bees.append(EmployeeBee(self.obj_function))

	def __initialize_onlookers(self):
		self.onlokeer_bees = []
		for itr in range(self.colony_size // 2):
			self.onlokeer_bees.append(OnLookerBee(self.obj_function))

	def __employee_bees_phase(self):
		_ = list(map(lambda bee: bee.explore(self.max_trials), self.employee_bees))

	def __calculate_probabilities(self):
		sum_fitness = sum(map(lambda bee: bee.get_fitness(), self.employee_bees))
		_ = list(map(lambda bee: bee.compute_prob(sum_fitness), self.employee_bees))

	def __select_best_food_sources(self):
		self.best_food_sources = list(filter(lambda bee: bee.prob > np.random.uniform(low=0, high=1), self.employee_bees))
		while not self.best_food_sources:
			self.best_food_sources = list(filter(lambda bee: bee.prob > np.random.uniform(low=0, high=1), self.employee_bees))

	def __onlooker_bees_phase(self):
		_ = list(map(lambda bee: bee.onlook(self.best_food_sources, self.max_trials), self.onlokeer_bees))

	def __scout_bees_phase(self):
		_ = list(map(lambda bee: bee.reset_bee(self.max_trials), self.onlokeer_bees + self.employee_bees))

	def optimize(self):
		self.__reset_algorithm()
		self.__initialize_employees()
		self.__initialize_onlookers()
		for itr in range(self.n_iter):
			self.__employee_bees_phase()
			self.__update_optimal_solution()

			self.__calculate_probabilities()
			self.__select_best_food_sources()

			self.__onlooker_bees_phase()
			self.__scout_bees_phase()

			self.__update_optimal_solution()
			self.__update_optimality_tracking()
			#if (itr % 10 == 0):
				#print("iter: {}: accuracy = {}".format(itr, "%04.03e" % self.optimal_solution.fitness))


def ColonyMask(x, y, colony_size, n_iters):
	colony = ArtificialBeeColony(MaskEvaluator(len(x[0]), x, y), colony_size=colony_size, n_iter=NIters, max_trials=MaxTrial)
	colony.optimize()
	return colony.optimal_solution.pos
	


x, y = LoadFeatures(all_features_file_name)


for fw in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
	for n in range(NRuns):
		FeatureWeight = fw
		mask = ColonyMask(x, y, ColonySize, NIters)


		accuracy = EvaluateMask(mask, x, y, feature_weight=0)
		print(str(n)+": Feature Weight = "+str(fw)+", accuracy = "+str(accuracy), flush = True)
