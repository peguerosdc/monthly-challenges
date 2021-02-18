"""
SOLUTION:
T = 8
m1 = 2,3,2
m2 = 7,1
"""

import pyquil.api as api
# import pyquil.api as WavefunctionSimulator
import numpy as np
from grove.pyqaoa.qaoa import QAOA
from pyquil.gates import *
from pyquil.paulis import PauliSum, PauliTerm, sI, sZ, sX, exponentiate_commuting_pauli_sum
import scipy
import math
from scipy.optimize import minimize
"""
STATE:
[m1, m2, ..., mN]

Where m1 = main chef queue = chef that will be the last to end

QUBITS:
0 = sí le toca hacer ese dish
1 = no le toca hacer ese dish
"""

# def qaoa_ansatz(betas, gammas, h_cost, h_driver):
#     pq = Program()
#     pq += [exponentiate_commuting_pauli_sum(h_cost)(g) + exponentiate_commuting_pauli_sum(h_driver)(b) for g,b in zip(gammas,betas)]
#     return pq

# def qaoa_cost(params, h_cost, h_driver, init_state_prog):
#     half = int(len(params)/2)
#     betas, gammas = params[:half], params[half:]
#     program = init_state_prog + qaoa_ansatz(betas, gammas, h_cost, h_driver)
#     return WavefunctionSimulator().expectation(prep_prog=program, pauli_terms=h_cost)


# create solver
class MinimumMakespanSolver(object):

    def __init__(self, queue, amount_chefs, steps=1):
        # build the list of qubits that will represent the state
        self.queue = queue
        self.N = len(queue)
        self.m = amount_chefs
        self.ancilla_per_chef = math.ceil(math.log2(sum(queue)))
        self.ancilla = (self.m-1)*self.ancilla_per_chef
        self.number_of_qubits = self.N*self.m + self.ancilla
        print(f"Building with {self.ancilla} ancilla qubits")
        # create weights of queue
        self.queue_weights = dict()
        for i,q in enumerate(sorted(queue)):
            self.queue_weights[q] = q*(1000**i)
        # Init params to run the QVM
        self.steps = steps
        self.betas = None
        self.gammas = None
        self.qaoa_inst = None
        self.qvm = api.QVMConnection()

    def create_cost_operators(self, weights):
        cost_operators = 0
        # Minimize the amount of dishes prepared my chef0 = chef_max
        cost_operators += self.create_cost_per_dish_operators(1)
        # Subject to:

        # i) no dish can be cooked more than once. This constraint
        # also enforces that each dish is cooked at least once
        # la mas ligera violacion de este constraint (E=w) tiene que ser
        # mucho mayor que el peor caso de la constraint principal
        # (que es cuando no minimiza nada y todas las tareas las hace
        # el primer chef)
        cost_operators += self.create_cost_simultaneous_chefs(weights[0])
        
        # ii) chef_max should take the highest amount of time
        cost_operators += self.create_cost_max_chef(weights[1])
        return [cost_operators]

    def create_cost_per_dish_operators(self, weight=None):
        weight = weight if weight else max(self.queue)
        return weight*self.cost_of_unprepared_dishes(chef=0, weighted=False)

    def create_cost_simultaneous_chefs(self, weight=None):
        # Estimate best weight considering this condition is definitely a no-go
        weight = weight if weight else sum(self.queue_weights.values())*200
        # Inhibit states where more than one chef is cooking the same dish
        cost_operators = 0
        for dish in range(self.N):
            op = -1*sI()
            for chef in range(self.m):
                op += self.get_bit(dish, chef)
            cost_operators += op**2
        return weight*cost_operators

    def create_cost_max_chef(self, weight=1):
        # estimate best weight
        weight = weight if weight else sum(self.queue_weights.values())*200
        # recalling that chef0 is the chef who is going to perform the max amount of dishes
        cost_operators = 0
        for chef in range(1,self.m):
            # get slack binary variables
            slack = sum( [ (2**i)*self.get_direct_bit(ancilla) for i,ancilla in enumerate(self.get_indexes_of_ancilla(chef))] )
            # get total hamiltonian
            cost_operators += (self.cost_of_unprepared_dishes(chef=0) - self.cost_of_unprepared_dishes(chef=chef) - slack)**2
        return weight*cost_operators

    def cost_of_unprepared_dishes(self, chef, weighted=False):
        cost_operators = 0
        for i,cost in enumerate(self.queue):
            weight = self.queue_weights[cost] if weighted else cost
            cost_operators += weight*self.get_bit(i, chef)
        return cost_operators

    def get_bit(self, dish, chef):
        """
        Operator that maps to 1 if the qubit is set, 0 otherwise:
          U|0> = 0
          U|1> = |1>
        """
        i = self.get_index(dish, chef)
        return self.get_direct_bit(i)

    def get_direct_bit(self, i):
        return .5 * (sI(i) - sZ(i))
        # return sX(i)

    def get_index(self,dish,chef):
        return self.N*chef + dish

    def solve(self, weight=None, samples=None):

        qubits = list(range(self.number_of_qubits))
        # Is how we encode our problem.
        # This is where all the information about our graph sits and it's encoded as a
        # combination of Pauli operators.
        cost_operators = self.create_cost_operators(weight)
        # Is how we encode what changes are possible in the realm of our problem.
        driver_operators = [-1*sum(sX(i) for i in range(self.number_of_qubits))]

        # build the QVM
        minimizer_kwargs = {'method': 'Nelder-Mead','options': {'ftol': 1.0e-2, 'xtol': 1.0e-2,'disp': False}}
        vqe_option = {'disp': None, 'return_all': True, 'samples': samples}
        self.qaoa_inst = QAOA(self.qvm, qubits, steps=self.steps, cost_ham=cost_operators,
                         ref_ham=driver_operators, store_basis=True,
                         minimizer=minimize,
                         minimizer_kwargs=minimizer_kwargs,
                         vqe_options=vqe_option,
                         rand_seed=42)
        #solve
        betas, gammas = self.qaoa_inst.get_angles()
        # probs = self.qaoa_inst.probabilities(np.hstack((betas, gammas)))
        most_frequent_string, sampling_results = self.qaoa_inst.get_string(betas, gammas, samples=1000)
        return sampling_results

    def get_indexes_of_ancilla(self, chef):
        chef = chef-1
        ancilla = [i for i in range(self.N*self.m + chef*self.ancilla_per_chef, self.N*self.m + (chef+1)*self.ancilla_per_chef) ]
        return ancilla


    def format_state(self, state):
        result = ""
        for chef in range(self.m):
            cooking = state[chef*self.N : chef*self.N + self.N]
            cooking = [self.queue[i]if c==1 else 0 for i,c in enumerate(cooking)]
            result += f"Chef {chef}: {cooking}. C={sum(cooking)}\n"
        return result

    def calculate_energy(self, state, weight=1):
        def ansatz(bits):
            ones = []
            for i,bit in enumerate(bits):
                if bit == "1":
                    ones.append(X(i))
                else:
                    ones.append(I(i))
            return Program(*ones)
        vqe_inst = VQE(minimizer=minimize, minimizer_kwargs={'method': 'nelder-mead'})
        # state to qubits
        state = "".join(f"{s}" for s in reversed(state))
        ans = ansatz(state)
        # calculate energy
        hamiltonians = [self.create_cost_per_dish_operators(1), self.create_cost_simultaneous_chefs(weight), self.create_cost_max_chef(weight)]
        # calculate
        return [vqe_inst.expectation(ansatz(state), h, None, self.qvm) if h else 0 for h in hamiltonians]

def plot(x,y):
    import matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=1)
    ax.set_ylabel('Probabily of (1,1,1,1,1)')
    ax.set_xlabel('Penalty weight')
    ax.set_title('Probability of the correct solution')
    plt.show()

def complies_1(queue, m, state):
    N = len(queue)
    cost_operators = 0
    for dish in range(N):
        op = -1
        for chef in range(m):
            op += state[N*chef + dish]
        cost_operators += op**2
    return cost_operators==0

def complies_2(queue, m, state):
    N = len(queue)
    # cost of chef 0:
    cost_0 = sum([ cost*state[N*0 + i] for i,cost in enumerate(queue) ])
    # recalling that chef0 is the chef who is going to perform the max amount of dishes
    cost_operators = []
    for chef in range(1,m):
        # get total hamiltonian
        cost_chef = sum([ cost*state[N*chef + i] for i,cost in enumerate(queue) ])
        cost_operators += [(cost_0 - cost_chef)**2]
    return cost_operators.count(0) == len(cost_operators)

if __name__ == '__main__':
    queue = [2,3,4]
    chefs = 2
    minimizer = MinimumMakespanSolver(queue=queue, amount_chefs=chefs)
    # find weights
    step = 0.02
    weight = [0,0]
    probabilities = []
    steps = range(30)
    for i in steps:
        print(f"Weight = {weight}")
        results = minimizer.solve(weight=weight)
        # filter results
        filtered = dict()
        total = 0
        for r in results.keys():
            key = r[0:len(queue)*chefs]
            count = filtered.get(key, 0) + results[r]
            filtered[key] = count
            total += count
        probabilities.append(filtered.get((1, 1, 0, 0, 0, 1), 0)/total)
        # display
        i = 0
        for state in dict(sorted(filtered.items(), key=lambda item: item[1],reverse=True)):
            # see if it complies with 1
            print(f"{state}: {filtered[state]/total}")
            print(f" - Complies with sim? {complies_1(queue, chefs, state)}")
            print(f" - Complies with min? {complies_2(queue, chefs, state)}")
            print(minimizer.format_state(state))
            if not complies_1(queue, chefs, state):
                weight[0] += step
            if not complies_2(queue, chefs, state):
                weight[1] += step
            # break
            i += 1
            if i==1:
                break
    plot(steps, probabilities)