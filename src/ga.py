import copy
from heapq import heappush, heappop, heapify
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math
from pprint import pprint
import sys

width = 200
height = 16

options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    "|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

# The level as a grid of tiles

class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    # Update this individual's estimate of its fitness.
    # This can be expensive so we do it once and then cache the result.
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Modify this, and possibly add more metrics.  You can replace this with whatever code you like.

        # Increased the weight of solvabilty
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=6.0
        )
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))

        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    def mutate(self, genome):
        # STUDENT implement a mutation operator, also consider not mutating this individual
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc

        parent_weights = {}


        # strategy for weighted mutations
        parent_weights['-'] = {}
        parent_weights['-'][(0, 2)] = 'o'
        parent_weights['-'][(3, 4)] = '?'

        parent_weights['E'] = {}
        parent_weights['E'][(0, 3)] = 'o'


        # sprinkle in a pit of random width 0 - 3 in a random place
        pit_width = random.randint(0, 5)
        pit_start_loc = random.randint(1, width - pit_width)
        for i in range(pit_start_loc, pit_start_loc + pit_width):
            genome[15][i] = '-'

        # #Place a random ramp going up or down of random height up to 4 blocks
        # Did not have enough time to finish this part up
        # prob_ramp = random.random()
        # # Generate some ramp
        # if prob_ramp <= 1:
        #     prob_direction = random.random()
        #     prob_height = random.randint(1, 4)
        #     if prob_direction <= 1:
        #         # make ramp that goes up
        #         start_pos = random.randint(1, width - 3 - prob_height)
        #
        #         can_make_ramp = True
        #         for i in range(start_pos, start_pos + prob_height + 1):
        #             if genome[15][i] == '-':
        #                 can_make_ramp = False
        #                 break
        #
        #         if can_make_ramp:
        #             for i in range(prob_height):
        #
        #                 for j in range(i + 1):
        #                     genome[14 - j][start_pos + i] = 'X'
        #
        #
        #     # else:
        #         #make ramp that goes down


        left = 1
        right = width - 1
        for row_index in range(height):
            for column_index in range(left, right):

                if column_index in list(range(width-3, width-1)):
                    continue

                try:
                    neighbor_above = genome[row_index - 1][column_index]
                except IndexError:
                    neighbor_above = -1

                try:
                    neighbor_below = genome[row_index + 1][column_index]
                except IndexError:
                    neighbor_below = -1

                try:
                    neighbor_left = genome[row_index][column_index - 1]
                except IndexError:
                    neighbor_left = -1

                try:
                    neighbor_right = genome[row_index][column_index + 1]
                except IndexError:
                    neighbor_right = -1

                try:
                    neighbor_top_left_diag = genome[row_index - 1][column_index - 1]
                except IndexError:
                    neighbor_top_left_diag = -1

                try:
                    neighbor_top_right_diag = genome[row_index - 1][column_index + 1]
                except IndexError:
                    neighbor_top_right_diag = -1

                try:
                    neighbor_bot_left_diag = genome[row_index + 1][column_index - 1]
                except IndexError:
                    neighbor_bot_left_diag = -1

                try:
                    neighbor_bot_right_diag = genome[row_index + 1][column_index + 1]
                except IndexError:
                    neighbor_bot_right_diag = -1

                tile = genome[row_index][column_index]


                if neighbor_below == -1:
                    if tile != 'X':
                        if tile != '-':
                            genome[row_index][column_index] = '-'
                            continue

                try:
                    weights_list = list(parent_weights[tile].keys())

                except:
                    weights_list = []

                chance = random.randint(1, 100) # Get a chance out of a 100 that will be a bound for executing.

                for weight_range in weights_list:
                    if chance in list(range(weight_range[0], weight_range[1] + 1)):
                        genome[row_index][column_index] = parent_weights[tile][weight_range]
                        break

                # removals for if floor section is now a pit
                if tile == '-' and neighbor_below == -1:
                    temp_row_index = row_index

                    # remove floating pipes
                    while neighbor_above == '|' or neighbor_above == 'T':
                        genome[temp_row_index][column_index] = '-'
                        temp_row_index -= 1
                        neighbor_above = genome[temp_row_index][column_index]

                    # remove anything else that was over that section
                    genome[row_index - 1][column_index] = '-'

                if neighbor_below == -1:
                    if tile != 'X' or tile != '-':
                        genome[row_index][column_index] = '-'

        return genome

    # Create zero or more children from self and other
    def generate_children(self, other):
        new_genome_1 = copy.deepcopy(self.genome)
        new_genome_2 = copy.deepcopy(other.genome)
        # Leaving first and last columns alone...
        # do crossover with other
        left = 1
        right = width - 1
        midpoint = random.randint(left, right)
        for row in range(height):
            for column in range(midpoint):
                new_genome_1[row][column] = other.genome[row][column]
                new_genome_2[row][column] = self.genome[row][column]


                # STUDENT Which one should you take?  Self, or other?  Why?
                # STUDENT consider putting more constraints on this to prevent pipes in the air, etc


        # do mutation
        new_genome_1 = self.mutate(new_genome_1)
        new_genome_2 = self.mutate(new_genome_2)

        return (Individual_Grid(new_genome_1), Individual_Grid(new_genome_2))

    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        return self.genome

    # These both start with every floor tile filled with Xs
    # STUDENT Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-3] = "v"
        for row in range(8, 14):
            g[row][-3] = "f"
        for row in range(14, 16):
            g[row][-3] = "X"

        return cls(g)

    @classmethod
    def random_individual(cls):
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        g = [random.choices(options, k=width) for row in range(height)]

        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-3] = "v"
        g[8:14][-1] = ["f"] * 6
        g[14:16][-1] = ["X", "X"]

        # Make sure pipe lines are valid and add their tops
        for column_index in range(width):
            highest_point = -1
            # Go from base up

            for row_index in range(height - 2, -1, -1):
                if g[row_index][column_index] != "|":

                    if g[row_index][column_index] != "m":
                        if highest_point != -1:
                            g[row_index][column_index] = "T"
                    break
                highest_point = row_index - 1

            if highest_point == -1:
                highest_point = height - 2
            for temp_row_index in range(highest_point , -1, -1):
                if g[temp_row_index][column_index] == '|':
                    g[temp_row_index][column_index] = '-'

        map_isclean = False
        while not map_isclean:
            for row_index in range(height):
                for column_index in range(width):
                    try:
                        neighbor_above = g[row_index - 1][column_index]
                    except IndexError:
                        neighbor_above = -1

                    try:
                        neighbor_below = g[row_index + 1][column_index]
                    except IndexError:
                        neighbor_below = -1

                    try:
                        neighbor_left = g[row_index][column_index - 1]
                    except IndexError:
                        neighbor_left = -1

                    try:
                        neighbor_right = g[row_index][column_index + 1]
                    except IndexError:
                        neighbor_right = -1

                    try:
                        neighbor_top_left_diag = g[row_index - 1][column_index - 1]
                    except IndexError:
                        neighbor_top_left_diag = -1

                    try:
                        neighbor_top_right_diag = g[row_index - 1][column_index + 1]
                    except IndexError:
                        neighbor_top_right_diag = -1

                    try:
                        neighbor_bot_left_diag = g[row_index + 1][column_index - 1]
                    except IndexError:
                        neighbor_bot_left_diag = -1

                    try:
                        neighbor_bot_right_diag = g[row_index + 1][column_index + 1]
                    except IndexError:
                        neighbor_bot_right_diag = -1


                    # Troll player by placing coins behind the end goal
                    if column_index == width - 1 or column_index == width - 2:
                        if row_index != height - 1:
                            g[row_index][column_index] = 'o'
                            continue

                    # Make sure there is nothing besides '-' above the goal
                    if column_index == width - 3:
                        if row_index < 7:
                            g[row_index][column_index] = '-'
                            continue

                    if column_index == width - 3:
                        if row_index > 7:
                            g[row_index][width - 3] = 'f'
                            continue



                    # check whether pipe top is valid
                    if g[row_index][column_index] == "T":
                        if neighbor_below != "|":
                            g[row_index][column_index] = "-"
                            continue

                    # check that enemy is not floating in the air
                    if g[row_index][column_index] == "E":
                        if neighbor_below not in ["X", "?", "M", "B", "T"]:
                            g[row_index][column_index] = "-"
                            continue


                    # make sure X is only the ground
                    if g[row_index][column_index] == "X":
                        if neighbor_below != -1:
                            g[row_index][column_index] = '-'
                            continue


            map_isclean = True

        return cls(g)



def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

# Inspired by https://www.researchgate.net/profile/Philippe_Pasquier/publication/220867545_Towards_a_Generic_Framework_for_Automated_Video_Game_Level_Creation/links/0912f510ac2bed57d1000000.pdf


class Individual_DE(object):
    # Calculating the level isn't cheap either so we cache it too.
    __slots__ = ["genome", "_fitness", "_level"]

    # Genome is a heapq of design elements sorted by X, then type, then other parameters
    def __init__(self, genome):
        self.genome = list(genome)
        heapify(self.genome)
        self._fitness = None
        self._level = None

    # Calculate and cache fitness
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Add more metrics?
        # STUDENT Improve this with any code you like

        # We put a higher weight on solvabiliy
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=5.0
        )
        penalties = 0
        # STUDENT For example, too many stairs are unaesthetic.  Let's penalize that
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) > 5:
            penalties -= 2
        # STUDENT If you go for the FI-2POP extra credit, you can put constraint calculation in here too and cache it in a new entry in __slots__.
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients)) + penalties
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        # STUDENT How does this work?  Explain it in your writeup.
        # STUDENT consider putting more constraints on this, to prevent generating weird things
        if random.random() < 0.1 and len(new_genome) > 0:
            to_change = random.randint(0, len(new_genome) - 1)
            de = new_genome[to_change]
            new_de = de
            x = de[0]
            de_type = de[1]
            choice = random.random()
            if de_type == "4_block":
                y = de[2]
                breakable = de[3]
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    breakable = not de[3]
                new_de = (x, de_type, y, breakable)
            elif de_type == "5_qblock":
                y = de[2]
                has_powerup = de[3]  # boolean
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    has_powerup = not de[3]
                new_de = (x, de_type, y, has_powerup)
            elif de_type == "3_coin":
                y = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                new_de = (x, de_type, y)
            elif de_type == "7_pipe":
                h = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    h = offset_by_upto(h, 2, min=2, max=height - 4)
                new_de = (x, de_type, h)
            elif de_type == "0_hole":
                w = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    w = offset_by_upto(w, 4, min=1, max=width - 2)
                new_de = (x, de_type, w)
            elif de_type == "6_stairs":
                h = de[2]
                dx = de[3]  # -1 or 1
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    h = offset_by_upto(h, 8, min=1, max=height - 4)
                else:
                    dx = -dx
                new_de = (x, de_type, h, dx)
            elif de_type == "1_platform":
                w = de[2]
                y = de[3]
                madeof = de[4]  # from "?", "X", "B"
                if choice < 0.25:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.5:
                    w = offset_by_upto(w, 8, min=1, max=width - 2)
                elif choice < 0.75:
                    y = offset_by_upto(y, height, min=0, max=height - 1)
                else:
                    madeof = random.choice(["?", "X", "B"])
                new_de = (x, de_type, w, y, madeof)
            elif de_type == "2_enemy":
                pass
            new_genome.pop(to_change)
            heappush(new_genome, new_de)
        return new_genome

    def generate_children(self, other):
        # STUDENT How does this work?  Explain it in your writeup.
        pa = random.randint(0, len(self.genome) - 1) if len(self.genome) > 0 else 0
        pb = random.randint(0, len(other.genome) - 1) if len(other.genome) > 0 else 0
        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []
        ga = a_part + b_part
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part
        # do mutation
        return Individual_DE(self.mutate(ga)), Individual_DE(self.mutate(gb))

    # Apply the DEs to a base level.
    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                # de: x, type, ...
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), random.choice([-1, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
        ]) for i in range(elt_count)]

        return Individual_DE(g)


Individual = Individual_Grid


def generate_successors(population):
    results = []

    elitist_results = []
    roulette_results = []

    # contains genome objects sorted in descended order by fitness
    genome_obj_list = []
    fitnessList=[]
    overallFitness=0

    #elitist
    for genome_obj in population:
        genome_obj.calculate_fitness()
        if genome_obj.fitness()!=0:
            overallFitness+=genome_obj.fitness()
        genome_obj_list.append(genome_obj)
    genome_obj_list = sorted(genome_obj_list, key=lambda obj: obj.fitness())[::-1]

    for i in range(0, 22):
        if len(genome_obj_list) == 0:
            break
        elitist_results.append(genome_obj_list.pop())

    #roulette
    for genome_obj in population:
        percent=genome_obj.fitness()/overallFitness
        fitnessList.append((percent, genome_obj))

    sorted_fitness_list = sorted(fitnessList, key=lambda obj: obj[0])[::-1]
    for item in sorted_fitness_list:
        itemChance = item[0]*100000
        chanceChoice = random.randint(1, 1000)
        if chanceChoice <= itemChance:
            roulette_results.append(item[1])


    for roulette_index in range(0, 22):
        if roulette_index >= len(roulette_results):
            break
        roulette_genome = roulette_results[roulette_index]

        for elitist_index in range(0, 22):
            if elitist_index >= len(elitist_results):
                break
            elitist_genome = elitist_results[elitist_index]

            result_tuple = elitist_genome.generate_children(roulette_genome)
            results.append(result_tuple[0])
            results.append(result_tuple[1])


    # STUDENT Design and implement this
    # Hint: Call generate_children() on some individuals and fill up results.
    return results


def ga():
    # STUDENT Feel free to play with this parameter
    pop_limit = 480
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization

        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
                      for _g in range(pop_limit)]

        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        population = pool.map(Individual.calculate_fitness,
                              population,
                              batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")


        try:
            while True:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")

                generation += 1
                # STUDENT Determine stopping condition
                stop_condition = False
                if stop_condition:
                    break
                # STUDENT Also consider using FI-2POP as in the Sorenson & Pasquier paper
                gentime = time.time()
                next_population = generate_successors(population)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # Calculate fitness in batches in parallel
                next_population = pool.map(Individual.calculate_fitness,
                                           next_population,
                                           batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pass
    return population


if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # STUDENT You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")
