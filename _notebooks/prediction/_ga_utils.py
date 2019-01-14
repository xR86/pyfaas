from __future__ import division, print_function

try:
    input = raw_input
except NameError:
    pass

import random

import pathos.multiprocessing as mp

from collections import Sequence
from collections import MutableSequence

from types import FunctionType

import deap
from deap import gp
from deap import base
from deap import tools

import numpy as np


def mutate_random(ind, low, up):
    """Randomizes a random chromosome of the individual.

    Args:
        ind (MutableSequence of float): The individual to mutate.
        low (Sequence of float)       : The lower bound of the interval for the new value of the chromosome.
        up  (Sequence of float)       : The lower upper of the interval for the new value of the chromosome.

    Returns:
        MutableSequence: The new individual.
    """

    gene_ind = random.randint(0, len(ind) - 1)
    low      = low[gene_ind]
    up       = up[gene_ind]

    ind[gene_ind] = random.uniform(low, up)
    return ind


def mutate_random_coeff_near(ind, low, up, coeffs):
    """Multiplies a random chromosome with a random coefficient from the coefficient interval.

    Args:
        ind    (MutableSequence of float): The individual to mutate.
        low    (Sequence of float)       : The lower bound of the interval for the new value of the chromosome.
        up     (Sequence of float)       : The lower upper of the interval for the new value of the chromosome.
        coeffs (pair of float)           : A sequence of 2 floats representing an interval.

    Returns:
        MutableSequence: The new individual.
    """

    gene_ind = random.randint(0, len(ind) - 1)
    low      = low[gene_ind]
    up       = up[gene_ind]

    coeff_low = coeffs[0]
    coeff_up  = coeffs[1]

    ind[gene_ind] = min(max(ind[gene_ind] * random.uniform(coeff_low, coeff_up), low), up)
    return ind


def mutate_random_coeff_far(ind, low, up, coeffs_low, coeffs_high):
    """Multiplies a random chromosome with a random coefficient from either the high or low coefficient intervals.

    Args:
        ind         (MutableSequence of float): The individual to mutate.
        low         (Sequence of float)       : The lower bound of the interval for the new value of the chromosome.
        up          (Sequence of float)       : The lower upper of the interval for the new value of the chromosome.
        coeffs_low  (pair of float)           : A sequence of 2 floats representing an interval.
        coeffs_high (pair of float)           : A sequence of 2 floats representing an interval.

    Returns:
        MutableSequence: The new individual.
    """

    gene_ind = random.randint(0, len(ind) - 1)
    low      = low[gene_ind]
    up       = up[gene_ind]

    coeffs    = coeffs_low if random.random < 0.5 else coeffs_high
    coeff_low = coeffs[0]
    coeff_up  = coeffs[1]

    ind[gene_ind] = min(max(ind[gene_ind] * random.uniform(coeff_low, coeff_up), low), up)
    return ind


class FitnessMax(base.Fitness):
    weights = (+1.0,)


class Individual(gp.PrimitiveTree):
    def __init__(self, content):
        super().__init__(content)
        self.fitness = FitnessMax()


def select_elite(population, elitism):
    """Returns the best individuals in the population along with the population after their removal.

    Args:
        population (Sequence of any): A sequence of individuals that have a `fitness` attribute that is a
                                      `deap.base.Fitness` object.
        elitism    (int)            : The number of elite individuals to select.

    Returns:
        Sequence of any: The selected elite.
        Sequence of any: The remaining population.
    """

    elite = []
    population = population[:]
    for i in range(elitism):
        best = max(list(range(len(population))), key = lambda i: population[i].fitness)
        elite += [population[i]]
        population.pop(best)

    return elite, population


def genetic_algorithm(toolbox, **kwargs):
    """Runs a genetic algorithm.

    Args:
        toolbox (deap.base.Toolbox): A DEAP toolbox containing the following registered functions:
                                     pop_init:  Function used to initialize the population. Should take one keyword
                                                parameter, `n`, an integer denoting the size of the population to be
                                                created. Will be called like `toolbox.pop_init(n = POPSIZE)`.
                                     evaluate:  Function used to evaluate an individual. Should take one argument, an
                                                individual created by `pop_init`, and return a number or tuple of
                                                numbers. Will be called like `toolbox.evaluate(ind)`.
                                     mate:      Function used to mate individuals in the operator phase. Should take
                                                2 arguments, 2 individuals created by `pop_init`, and change their
                                                internal state (attributes). No return value is needed. Will be
                                                called like `toolbox.mate(ind1, ind2)`.
                                     mutations: List of functions that will be used as mutations. Each of the
                                                functions should take one argument, an individual created by
                                                `pop_init`, and and change its internal state (attributes). No return
                                                value is needed. If only one mutation is needed simply provide a list
                                                containing only one function. Will be called like
                                                `toolbox.mutations[i](ind)`.

    Keyword Args:
        cxpb         (float)         : Probability of crossover happening to a pair of individuals during the operator
                                       phase. Defaults to `0.25`.
        mutpb        (float)         : Probability of mutation happening to an individual during the operator phase.
                                       Defaults to `0.25`.
        mutpbdist    (list of float) : Tuple of probabilities of each mutation provided in `mutations` being selected as
                                       the mutation to be performed on a particular selected individual. The sum of
                                       these probabilities should be 1.0, as with any distribution. Defaults to a
                                       uniform distribution where the chance for each mutation to be selected starts
                                       from `1/num_possible_mutations`.
        mutpbdistinc (float)         : By how much to increase the most successful mutation each generation (the
                                       mutation that yielded the best average increase in fitness). Defaults to `0.0`
                                       (no increase).
        mutpbdistmin (float)         : The minimum value a probability of a certain mutation mai decrease to. Defaults
                                       to `0.5/(num_possible_mutations-1)` if there is more than one mutation and `0.0`
                                       if there is only one mutation.
        popsize      (int)           : Number of individuals to create an maintain throughout the algorithm. Defaults to
                                       `100`.
        popreplace   (float)         : How much of the population to completely discard and re-initialize at every
                                       generation. This coefficient is applied *after* the elite individuals from
                                       elitism have been removed from the general population. Defaults to `0.1`
        tournsize    (int)           : The number of a tournament. Used in the tournament selection. Defaults to
                                       `int(popsize/5.0)`
        gens         (int)           : Number of generations to run for. Defaults to `100`
        addgens      (bool)          : Whether to present the user with a prompt allowing for adding additional
                                       generations at the end of the algorithm. Useful in cases where consistent
                                       improvement is still observed towards the end of the algorithm and future
                                       improvement is suspected to still exist. Does nothing if `verbose` is not `True`.
                                       Defaults to `False`.
        fitweight    (tuple of float): Tuple of weights for each of the elements in the fitness vector returned by
                                       `evaluate`. In case optimization of a function with one output is desired simply
                                       pass a tuple with one element (usually (+1.0,) or (-1.0,) depending on whether
                                       maximization or minimization is desired. Defaults to `(+1.0,)`.
        elitism      (int)           : The top `elitism` individuals of each generation will be saved each generation.
                                       Defaults to `0`.
        njobs        (int)           : The maximum number of processes to use to paralellize the fitness computation.
                                       Defaults to `1`.
        hof          (HallOfFame)    : `deap.tools.HallOfFame` instance. Defaults to `None`.
        stats        (Statistics)    : `deap.tools.Statistics` instance. Defaults to `None`.
        verbose      (bool)          : Enable output. Defaults to `False`.
        dispfreq     (int)           : If verbose is enabled, set the frequency with which output is provided. E.g. if
                                       `dispfreq` is `3` then detailed output will be provided once every 3 generations.
                                       Defaults to `1`.
        dispfunc     (FunctionType)  : Function to be called every `dispfreq` generations in case `verbose` is `True`.
                                       The function will take the following arguments:
                                       population   (Sequence of any): The population of the current generation.
                                       logbook      (Logbook)        : `deap.tools.Logbook` instance recording any stats
                                                                       given trough the stats argument. To print the
                                                                       regular logbook output use
                                                                       `print(logbook.stream)`. See `Logbook.select()`.
                                       hall_of_fame (HallOfFame)     : `deap.tools.HallOfFame` instance given to the
                                                                       original `genetic_algorithm` call.
                                       This function can be used for custom output or plotting. Do not modify any of the
                                       arguments received as that may also modify the actual population maintained by
                                       the algorithm. Defaults to `lambda *_, **__: None`.

    Returns:
        Any    : The fittest individual from the last generation.
        Logbook: The `deap.tools.Logbook` that recorded `stats` during the algorithm.
    """

    # Argument parsing

    CXPB  = kwargs.get('cxpb',  0.25)
    MUTPB = kwargs.get('mutpb', 0.25)

    MUTPB_DIST          = kwargs.get('mutpbdist', list(map(lambda _: 1.0 / len(toolbox.mutations),
                                                           toolbox.mutations)))
    MUTPB_DIST_INCREASE = kwargs.get('mutpbdistinc', 0.0)
    MUTPB_DIST_MIN      = kwargs.get('mutpbdistmin', ((0.5 / (len(toolbox.mutations) - 1))
                                                      if len(toolbox.mutations) > 1
                                                      else 0.0))

    POPSIZE         = kwargs.get('popsize',    100)
    POPREPLACECOEFF = kwargs.get('popreplace', 0.1)

    TOURNSIZE = kwargs.get('tournsize', int(POPSIZE / 5.0))

    GENERATIONS                = kwargs.get('gens',    100)
    ADD_GENERATIONS_AT_RUNTIME = kwargs.get('addgens', False)

    FITNESS_WEIGHTS = kwargs.get('fitweight', (+1.0,))

    ELITISM = kwargs.get('elitism', 0)

    VERBOSE           = kwargs.get('verbose',  False)
    DISPLAY_FREQUENCY = kwargs.get('dispfreq', 1)

    pool = kwargs.get('pool')

    user_display_func = kwargs.get('dispfunc', lambda *_, **__: None)

    hall_of_fame = kwargs.get('hof', None)
    stats = kwargs.get('stats', None)

    # Checks

    assert 0.0 <= CXPB  <= 1.0, 'Crossover probability must be in the interval [0.0, 1.0].'
    assert 0.0 <= MUTPB <= 1.0, 'Mutation probability must be in the interval [0.0, 1.0].'

    assert sum(MUTPB_DIST) == 1.0, 'Mutation probability distribution must have a sum of probabilities of 1.0.'
    assert 0.0 <= MUTPB_DIST_INCREASE <= 1.0, 'Mutation probability increase must be in the interval [0.0, 1.0].'
    assert 0.0 <= MUTPB_DIST_MIN <= 1.0, 'Min mutation probability must be in the interval [0.0, 1.0].'

    assert POPSIZE > 0, 'Population size must be positive.'
    assert 0.0 <= POPREPLACECOEFF <= 1.0, 'Population replacement coefficient must be in the interval [0.0, 1.0].'

    assert TOURNSIZE > 0, 'Tournament size must be positive.'
    assert TOURNSIZE < POPSIZE, 'Tournament size must be less than population size.'

    assert GENERATIONS > 0, 'Number of generations must be positive.'

    assert ELITISM >= 0, 'Elitism must be positive or 0.'

    # Constant processing

    MUTPB_DIST_DECREASE = ((MUTPB_DIST_INCREASE / (len(toolbox.mutations) - 1))
                           if len(toolbox.mutations) > 1
                           else 0.0)

    MUTPB_DIST_MAX = 1.0 - MUTPB_DIST_MIN * (len(toolbox.mutations) - 1)

    POPREPLACE = int((POPSIZE - ELITISM) * POPREPLACECOEFF)
    POPKEEP = (POPSIZE - ELITISM) - POPREPLACE

    MULTI_OBJECTIVE = len(FITNESS_WEIGHTS) > 1


    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Main algorithm

    if VERBOSE:
        print('Running GA...\n')

    min_fits, mean_fits, max_fits = [], [], []

    mutation_diff_lists = [[] for _ in toolbox.mutations]
    mutation_diff_avgs  = [0 for _ in toolbox.mutations]

    population = toolbox.pop_init(n = POPSIZE)

    evals_needed = sum(not individual.fitness.valid for individual in population)
    fitnesses = list(pool.map(toolbox.evaluate, population))

    for individual, fitness in zip(population, fitnesses):
        individual.fitness.values = fitness

    try:
        generation_number = 0
        while generation_number < GENERATIONS:
            generation_number += 1

            # Gather and display statistics

            if hall_of_fame is not None:
                hall_of_fame.update(population)

            record = stats.compile(population) if stats else {}
            logbook.record(gen = generation_number, nevals = evals_needed, **record)

            if VERBOSE and generation_number % DISPLAY_FREQUENCY is 0:
                user_display_func(population,
                                  logbook,
                                  hall_of_fame)

            # Select the surviving offspring and refill the population

            elite, non_elite = select_elite(population, ELITISM)

            offspring = elite + toolbox.select(non_elite, POPKEEP) + toolbox.pop_init(n = POPREPLACE)
            offspring = [toolbox.clone(child) for child in offspring]

            # Apply operators (AND)

            evals_needed = 0
            mutation_diff_lists = [[] for _ in toolbox.mutations]
            mutation_diff_avgs  = [0 for _ in toolbox.mutations]

            random.shuffle(offspring)

            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                if random.random() <= CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

                for child in (child1, child2):

                    if random.random() <= MUTPB:

                        mutation_type = random.random()

                        evals_needed += not child.fitness.valid
                        old_fitness = (child.fitness.values
                                       if child.fitness.valid
                                       else toolbox.evaluate(child))
                        del child.fitness.values

                        for i, threshold in enumerate(np.cumsum(MUTPB_DIST)):
                            if mutation_type <= threshold:

                                toolbox.mutations[i](child)

                                evals_needed += 1
                                new_fitness = toolbox.evaluate(child)
                                child.fitness.values = new_fitness

                                fitness_diff = new_fitness[0] - old_fitness[0]
                                mutation_diff_lists[i] += [fitness_diff]

                                break

            # Operator adjustment

            mutation_diff_avgs = [np.average(mutation_diffs) if mutation_diffs else 0
                                  for mutation_diffs
                                  in mutation_diff_lists]

            best_diff_ind = np.argmax(mutation_diff_avgs)

            for i in range(len(MUTPB_DIST)):
                if i == best_diff_ind:
                    MUTPB_DIST[i] = min(MUTPB_DIST_MAX, MUTPB_DIST[i] + MUTPB_DIST_INCREASE)
                else:
                    MUTPB_DIST[i] = max(MUTPB_DIST_MIN, MUTPB_DIST[i] - MUTPB_DIST_DECREASE)

            # Evaluate unevaluated offspring

            evals_needed += sum(not individual.fitness.valid for individual in population)
            fitnesses = list(pool.map(lambda child: (child.fitness.values
                                                     if child.fitness.valid
                                                     else toolbox.evaluate(child)),
                                      offspring))

            for child, fitness in zip(offspring, fitnesses):
                child.fitness.values = fitness

            # Replace the old population

            population[:] = offspring

            if generation_number == GENERATIONS:

                fittest_individual = max(population, key = lambda ind: sum(ind.fitness.wvalues))

                if VERBOSE:
                    print('\n'
                          '{} generations done.\n'
                          '\n'
                          'Fittest individual: {}\n'.format(generation_number,
                                                            fittest_individual))

                    if ADD_GENERATIONS_AT_RUNTIME:
                        while True:
                            if input('Keep running? (y/n) ') == 'y':
                                try:
                                    GENERATIONS += int(input('How many more generations? '))
                                    print()
                                    break
                                except ValueError:
                                    print('\nInvalid number.\n')
                                    continue
                            else:
                                break
                    else:
                        return fittest_individual, logbook
                else:
                    return fittest_individual, logbook

    except KeyboardInterrupt:
        print('\nManual exit.\n')


def hill_climbing(evaluate, initial_solution, **kwargs):
    """Optimizes the `evaluate` function using the SAHC algorithm.

    Splits the coordinate domains into discrete steps equal to `discretization_percent` % of the entire domain, capped
    at the lower end at `resolution`, then uses the computed steps to perform SAHC.

    When there are no other improvements, the step of the first coordinate is divided by 10 (lowered by an order of
    magnitude) and the SAHC attempts to find new improvements. If none are found, then the step of the second coordinate
    is lowered by an order of magnitude and again the SAHC attempts to find new improvements and so on.

    After all of the coordinates have been lowered by an order of magnitude, the first coordinate is lowered again and
    so on. This process caps when all of the steps are equal to `resolution`.

    Args:
        evaluate         (function returning float): The function to optimize.
        initial_solution (MutableSequence of float): The solution to start from.

    Keyword Args:
        lb                        (Sequence of float): The upper bound of each coordinate of a candidate solution.
                                                       Defaults to a Sequence of -2^32.
        ub                        (Sequence of float): The lower bound of each coordinate of a candidate solution.
                                                       Defaults to a Sequence of 2^32 - 1.
        resolution                (Sequence of float): The resolution (or "step") of each coordinate. Defaults to a
                                                       Sequence of 10^(-16).
        discretization_percent    (float)            : The interval [`lb`, 'ub'] will be discretized in
                                                       100/`discretization_percent` "steps", each representing
                                                       `discretization_percent` % of the entire interval. Defaults to
                                                       1.0.
        iterations                (int)              : The number of iterations to run. Defaults to infinity.
        add_iterations_at_runtime (bool)             : Whether or not to ask the user if / how many more iterations
                                                       should be ran then the given number of iterations is completed.
                                                       Defaults to `False`.
        verbose                   (bool)             : Enable verbose mode. Defaults to `False`.

    Returns:
            MutableSequence of float: The best argument found for `evaluate`.
            MutableSequence of float: The values of `evaluate` for the best neighbour of each iteration.
    """

    lb = kwargs.get('lb', [- (2 ** 32) for _ in initial_solution])
    ub = kwargs.get('lb', [(2 ** 32) - 1 for _ in initial_solution])

    resolution = kwargs.get('resolution', [10 ** -16 for _ in initial_solution])

    discretization_percent = kwargs.get('discretization_percent', 1.0)

    iterations = kwargs.get('iterations', float('inf'))

    add_iterations_at_runtime = kwargs.get('add_iterations_at_runtime', False)

    verbose = kwargs.get('verbose', False)

    if verbose:
        print('Running SAHC...\n')

    iteration = 0
    best_sol = initial_solution

    if verbose:
        print(
            'Coordinate bounds initially discretized in {} % intervals.\n'.format(
                discretization_percent * 100
            )
        )

    next_coord_ind = 0

    adjusted_resolution = [
        max(
            resolution[coord_ind],
            (ub[coord_ind] - lb[coord_ind]) * discretization_percent
        )
        for coord_ind
        in range(len(resolution))
    ]

    hc_fits = []

    best_sol_fitness = evaluate(best_sol)

    while True:
        try:
            iteration += 1

            best_sol_fitness = evaluate(best_sol)

            hc_fits += [best_sol_fitness]

            neighbourhood = []
            fitnesses = []

            for coord_ind in range(len(best_sol)):
                neighbour_up = best_sol[:]
                neighbour_up[coord_ind] += adjusted_resolution[coord_ind]

                neighbourhood += [neighbour_up]
                fitnesses += [evaluate(neighbourhood[-1])]

                neighbour_down = best_sol[:]
                neighbour_down[coord_ind] -= adjusted_resolution[coord_ind]

                neighbourhood += [neighbour_down]
                fitnesses += [evaluate(neighbourhood[-1])]

            best_neighbour_ind = min(range(len(neighbourhood)), key = lambda ind: fitnesses[ind])

            if verbose:
                print(
                    'Iteration {:5d}\t|\t'.format(iteration),
                    'Current fitness: {:12.6f} MSE\t|\t'.format(best_sol_fitness),
                    'Best neighbour: {:12.6f} MSE\t|\t'.format(fitnesses[best_neighbour_ind]),
                    'Improvement: {:12.6f} MSE'.format(fitnesses[best_neighbour_ind] - best_sol_fitness)
                )

            if fitnesses[best_neighbour_ind] < best_sol_fitness:
                best_sol = neighbourhood[best_neighbour_ind]
            else:
                if adjusted_resolution == resolution:
                    break
                else:
                    if verbose:
                        print(
                            '\n'
                            'Peak found for discretization of coordinate {} bounds in {} % '
                            'intervals.\n'
                            '\n'
                            'Best solution: {} \n'
                            '\n'
                            'Increased discretization to {} % for coordinate {}. \n'.format(
                                next_coord_ind,
                                discretization_percent * 100,
                                best_sol,
                                discretization_percent * 10,
                                next_coord_ind
                            )
                        )

                    adjusted_resolution[next_coord_ind] = max(
                        adjusted_resolution[next_coord_ind] / 10.0,
                        resolution[next_coord_ind]
                    )

                    next_coord_ind = (next_coord_ind + 1) % len(resolution)

                    if next_coord_ind == 0:
                        discretization_percent /= 10.0

            if iteration == iterations:
                if verbose:
                    print(
                        '\n'
                        '{} iterations done.\n'
                        '\n'
                        'Best solution: {}\n'.format(
                            iteration,
                            best_sol
                        )
                    )
                if add_iterations_at_runtime:
                    while True:
                        if input('Keep running? (y/n) ') == 'y':
                            try:
                                iterations += int(input('How many more iterations? '))
                                print()
                            except ValueError:
                                print('\nInvalid number.\n')
                                continue
                        break
                else:
                    break

        except KeyboardInterrupt:
            break

    if verbose:
        print(
            'Peak: {:12.6f} MSE'.format(best_sol_fitness),
            '\n\n',
            'Peak coordinates: {}\n'.format(best_sol),
            sep = ''
        )

    return best_sol, hc_fits
