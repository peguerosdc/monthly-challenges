# Generate QUBO equation
def qubo(n):
    import itertools
    from sympy import symbols
    # create variables for n qubits
    q = symbols([f"q{i}" for i in range(n)])
    a = symbols([f"a{i}" for i in range(n)])
    b = symbols([f"b{i}{j}" for i,j in itertools.combinations(range(n),2)])
    # build H for each qubit
    expr = 0
    for qi,ai in zip(q,a):
        expr += qi*ai
    # build H for each combination
    for i,(x,y) in enumerate(itertools.combinations(q,2)):
        expr += b[i]*x*y
    return q,a,b,expr

def truth_table(n, alist=None, ising=False):
    import itertools
    table = []
    for combination in itertools.product([1,-1] if ising else [0,1],repeat=n):
        q,a,b,expr = qubo(n)
        expr = expr.subs([ (qi, combination[i]) for i,qi in enumerate(q)])
        # evaluate a if given
        if alist:
            expr = expr.subs([ (ai,alist[i]) for i,ai in enumerate(a) ])
        print(f"{combination} : {expr}")
        table.append((combination, expr))
    return table

def classic_solver(queue=[3,1], m=2):
    result = {'comb': [], 'cost':sum(queue)}
    import itertools
    n = len(queue)
    for state in itertools.product([0,1],repeat=n*m):
        # check there are no repeated jobs
        repeated = False
        for i in range(0,n):
            bits = [state[j] for j in range(i,len(state),n)]
            if bits.count(1) > 1:
                repeated = True
                break
        if not repeated and state.count(1)==n:
            # first chef must have the highest cost
            costs = []
            for chef in range(m):
                costs.append( sum([ a*b for a,b in zip(state[chef*n:chef*n + n], queue) ]) )
            is_first_max = True
            for cost in costs[1:]:
                if costs[0] < cost:
                    is_first_max = False
                    break
            # print(f"{state} , {costs}, {is_first_max}")
            # check if it is the most optimal
            if is_first_max:
                # print(f"{state} : {costs[0]}")
                if costs[0] == result['cost']:
                    result['comb'].append(state)
                elif costs[0] < result['cost']:
                    result['comb'] = [state]
                    result['cost'] = costs[0]
    return result

def format_state(state, m, N, queue):
    result = ""
    for chef in range(m):
        cooking = state[chef*N : chef*N + N]
        cooking = [queue[i] for i,c in enumerate(cooking) if c==1]
        result += f"Chef {chef} cooking {cooking} with cost {sum(cooking)} \n"
    return result

"""
Chef 0 cooking [2, 3] with cost 5 
Chef 1 cooking [4] with cost 4 
"""
m = 2
queue = [2,3,4]

results = classic_solver(queue,m)
for r in results['comb']:
    print(format_state(r,m,len(queue),queue))