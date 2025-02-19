import gurobipy as gp
from gurobipy import GRB
import numpy as np
from Clustering_Functions import HH_proxy, Borda_vector, HH_dist, Borda_dist

########################################
# CONTINUOUS
# MODELS
########################################

def continuous_model(ballots, weights, num_clusters, value_set):
    """
    Returns a Gurobi model for the continuous k-medians problem.

    Parameters:
    ballots: np.array, shape=(num_ballots, dimensions)
        The ballots to be clustered. These ballots are unique and their counts are specified by the weights.    
    weights: np.array, shape=(num_ballots,)
        Represents the amount of each ballot
    num_clusters: int
        The number of clusters to be formed
    value_set: np.array
        The set of possible values for each dimension of the ballots.
    """
    
    num_ballots, dimensions = ballots.shape
    assert weights.shape[0] == num_ballots, "Weights and ballots dimensions do not match."
    D = range(num_ballots)
    K = range(num_clusters)
    V = value_set
    Dims = range(dimensions)
    model = gp.Model("continuous_k_medians")

    # x[j,r]: 1 if ballot j is assigned to cluster r
    x = {(j,r): model.addVar(vtype=GRB.BINARY, name=f"x[{j},{r}]") 
         for j in D for r in K}
    
    # z[i,r,v]: 1 if ith dimension of the median of the rth cluster is v
    z = {(i,r,v): model.addVar(vtype=GRB.BINARY, name=f"z[{i},{r},{v}]") 
        for i in Dims for r in K for v in V}

    # W[i,r,v]: total weight of points in D assigned to cluster r with coordinate i of value v
    W = {(i,r,v): model.addVar(vtype=GRB.INTEGER, name=f"W[{i},{r},{v}]")
         for i in Dims for r in K for v in V}
                
    # C[i,r]: contribution of coordinate i to cost of cluster r
    # TODO: Integer or continuous?
    C = {(i,r): model.addVar(vtype=GRB.INTEGER, name=f"C[{i},{r}]")
        for i in Dims for r in K}
    
    # Define "big-M" (fixed to be max - min)
    M = np.sum(weights) * (np.max(value_set)-np.min(value_set)) 
    
    for j in D:
        # Constraint: Each ballot j is assigned to exactly one cluster r
        model.addLConstr(gp.quicksum([x[j,r] for r in K]) == 1)

    # break symmetry to have cluster r-1 have fewer distinct ballots than cluster r
    for r in range(num_clusters-1):
        model.addLConstr(sum(x[i,r] for i in range(num_ballots))<= sum(x[i,r+1] for i in range(num_ballots)))

    for i in Dims:
        for r in K:
            for v in value_set:
                model.addLConstr(gp.quicksum([weights[j]*x[j,r] for j in D if ballots[j, i] == v]) - W[i,r,v] == 0)
                
    
    for i in Dims:
        for r in K:
            # Constraint: There is exactly one median value for each cluster and coordinate
            model.addLConstr(gp.quicksum(z[i,r,v] for v in V) == 1)            
            for v_target in V:
                # Constraint: Median finding constraints
                model.addLConstr(gp.quicksum([W[i,r,v] if v > v_target else -1*W[i,r,v] for v in V]) - M*(1-z[i,r,v_target]) <= 0)
                model.addLConstr(gp.quicksum([W[i,r,v] if v < v_target else -1*W[i,r,v] for v in V]) - M*(1-z[i,r,v_target]) <= 0)

                # Constraint: Bound C
                model.addLConstr(C[i,r] - gp.quicksum([np.abs(v - v_target)*W[i,r,v] for v in V]) <= 0)
                model.addLConstr(gp.quicksum([np.abs(v - v_target)*W[i,r,v] for v in V]) - C[i,r] - M*(1-z[i,r,v_target]) <= 0)

                # TODO: Do these constraints produce the same results?
                # # Constraint: Bound C
                # model.addLConstr(C[i,r] - gp.quicksum([np.abs(v - v_target)*W[i,r,v] for v in V]) - M*(1-z[i,r,v_target]) <= 0)
                # model.addLConstr(gp.quicksum([np.abs(v - v_target)*W[i,r,v] for v in V]) - C[i,r] - M*(1-z[i,r,v_target]) <= 0)
    
    model.setObjective(gp.quicksum([C[i,r] for i in Dims for r in K]), GRB.MINIMIZE)

    # cost = model.addVar(vtype=GRB.INTEGER, name="cost")
    # model.addLConstr(cost - gp.quicksum([C[i,r] for i in Dims for r in K]) == 0)
    # model.setObjective(cost, GRB.MINIMIZE)

    model.update()
    return model


def continuous_bordaP(election_data, num_clusters):
    """
    Returns a Gurobi model for the continuous k-medians problem using pessimistic borda embeddings.
    Note: The current value set for n candidates is (0...n-1).

    Parameters:
    election_data: dict
        A dictionary matching ballots to weights.
    num_clusters: int
    """
    all_ballots = list(election_data.keys())
    num_ballots = len(all_ballots)
    num_cands = len(set([item for ranking in all_ballots for item in ranking]))

    ballots = np.array([Borda_vector(ballot, num_cands = num_cands, borda_style='pes') for ballot in all_ballots])
    weights = np.array([election_data[ballot] for ballot in all_ballots])
    value_set = np.array(range(num_cands))

    return continuous_model(ballots, weights, num_clusters, value_set)


def continuous_bordaA(election_data, num_clusters):
    """
    Returns a Gurobi model for the continuous k-medians problem using average borda embeddings.

    Parameters:
    election_data: dict
        A dictionary matching ballots to weights.
    num_clusters: int
    """
    all_ballots = list(election_data.keys())
    num_ballots = len(all_ballots)
    num_cands = len(set([item for ranking in all_ballots for item in ranking]))

    ballots = np.array([Borda_vector(ballot, num_cands = num_cands, borda_style = 'avg', start=1) for ballot in all_ballots])
    weights = np.array([election_data[ballot] for ballot in all_ballots])
    value_set = np.array([l/2 for l in range(1, 2*num_cands + 1)])

    return continuous_model(ballots, weights, num_clusters, value_set)


def continuous_hh(election_data, num_clusters):
    """
    Returns a Gurobi model for the continuous k-medians problem using head-to-head embeddings

    Parameters:
    election_data: dict
        A dictionary matching ballots to weights.
    num_clusters: int
    """
    all_ballots = list(election_data.keys())
    num_ballots = len(all_ballots)
    num_cands = len(set([item for ranking in all_ballots for item in ranking]))
    
    ballots = 2*np.array([HH_proxy(ballot,num_cands=num_cands) for ballot in all_ballots])
    weights = np.array([election_data[ballot] for ballot in all_ballots])
    value_set = np.array([-1, 0, 1])

    return continuous_model(ballots, weights, num_clusters, value_set)


########################################
# CONTINUOUS
# RESTRICTED
# MODELS
########################################


def continuous_rest_hh(election_data, num_clusters):
    """
    Returns a Gurobi model with centroids restricted to possible ballots using head-to-head embeddings.

    Parameters:
    election_data: dict
        A dictionary matching ballots to weights.
    num_clusters: int
    """
    all_ballots = list(election_data.keys())
    num_ballots = len(all_ballots)
    num_cands = len(set([item for ranking in all_ballots for item in ranking]))
    
    ballots = 2*np.array([HH_proxy(ballot,num_cands=num_cands) for ballot in all_ballots])
    weights = np.array([election_data[ballot] for ballot in all_ballots])
    value_set = np.array([-1, 0, 1])

    num_ballots, num_cand_choose_2 = ballots.shape
    assert weights.shape[0] == num_ballots, "Weights and ballots dimensions do not match."
    D = range(num_ballots)
    K = range(num_clusters)
    V = value_set
    Dims = range(num_cand_choose_2)
    model = gp.Model("continuous_rest_k_medians")

    # x[j,r]: 1 if ballot j is assigned to cluster r
    x = {(j,r): model.addVar(vtype=GRB.BINARY, name=f"x[{j},{r}]") 
         for j in D for r in K}
    
    # z[i,r,v]: 1 if ith dimension of the median of the rth cluster is v
    z = {(i,r,v): model.addVar(vtype=GRB.BINARY, name=f"z[{i},{r},{v}]") 
        for i in Dims for r in K for v in V}

    # W[i,r,v]: total weight of points in D assigned to cluster r with coordinate i of value v
    W = {(i,r,v): model.addVar(vtype=GRB.INTEGER, name=f"W[{i},{r},{v}]")
         for i in Dims for r in K for v in V}
                
    # C[i,r]: contribution of coordinate i to cost of cluster r
    # Has to be continuous since some values for borda are not integral
    C = {(i,r): model.addVar(vtype=GRB.CONTINUOUS, name=f"C[{i},{r}]")
        for i in Dims for r in K}
    
    # Define "big-M" ( is this good enough?)
    # Fixed M here
    M = np.sum(weights) * (np.max(value_set)-np.min(value_set)) 
    
    for j in D:
        # Constraint: Each ballot j is assigned to exactly one cluster r
        model.addLConstr(gp.quicksum([x[j,r] for r in K]) == 1)

    # break symmetry to have cluster r-1 have fewer distinct ballots than cluster r
    for r in range(num_clusters-1):
        model.addLConstr(sum(x[i,r] for i in range(num_ballots))<= sum(x[i,r+1] for i in range(num_ballots)))

    for i in Dims:
        for r in K:
            for v in value_set:
                model.addLConstr(gp.quicksum([weights[j]*x[j,r] for j in D if ballots[j, i] == v]) - W[i,r,v] == 0)
                
    
    for i in Dims:
        for r in K:
            # Constraint: There is exactly one median value for each cluster and coordinate
            model.addLConstr(gp.quicksum(z[i,r,v] for v in V) == 1)            
            for v_target in V:
                # Constraint: Ensure associated cost Cir is exactly the contribution of coordinate i to the cost of cluster r
                model.addLConstr(C[i,r] - gp.quicksum([np.abs(v - v_target)*W[i,r,v] for v in V]) - M*(1-z[i,r,v_target]) <= 0)
                model.addLConstr(gp.quicksum([np.abs(v - v_target)*W[i,r,v] for v in V]) - C[i,r] - M*(1-z[i,r,v_target]) <= 0)

    # Potential ballot constraints

    # n[l,r]: 1 iff the candidate l is not on the ballot corresponding to cluster r
    n = {(l,r): model.addVar(vtype=GRB.BINARY, name=f"n[{l},{r}]") for l in range(num_cands) for r in K}

    # p[l,m,r]: 1 iff ballot corresponding to cluster r has candidate l preferred to candidate m
    p = {(l,m,r): model.addVar(vtype=GRB.BINARY, name=f"p[{l},{m},{r}]") for l in range(num_cands) for m in range(num_cands) for r in K if l != m}

    # Converts coordinates to the candidate it corresponds to
    # So at the value at coord i == 1, we prefer candidate f(i) to candidate s(i)
    # Based off of hh_proxy which defines the hh embedding as follows
    # "This is a vector with one entry for each pair of candidates ordered in the natural way; namely {(1,2),(1,3),...,(1,n),(2,3),...}.  The entries lie in {-1/2,0,1/2} depending on whether the lower-indexed candidate {looses, ties, wins} the head-to-head comparison."
    # print(num_cands)
    f = [i for i in range(num_cands) for _ in range(i+1, num_cands)]
    s = [j for i in range(num_cands) for j in range(i+1, num_cands)]

    # Constraint (16): If we have a 0 value, then both the candidates it corresponds to must not be on the ballot.
    for r in K:
        for i in Dims:
            model.addLConstr(z[i,r,0] <= n[f[i],r])
            model.addLConstr(z[i,r,0] <= n[s[i],r])
    
    for r in K:
        for l in range(num_cands):
            for m in range(num_cands):
                if l != m:
                    # Constraint (17): If l is not on the ballot, it can't be preferred to another candidate.
                    # Fixed
                    model.addLConstr(n[l,r] <= 1 - p[l,m,r])
                    for t in range(num_cands):
                        if t != l and t != m:
                            # Constraint (18): Preference relation is transitive
                            model.addLConstr(p[l,m,r] + p[m,t,r] <= 1 + p[l,t,r])
    
    for r in K:
        for i in Dims:
            # Constraints (19) and (20): If the value of coord i is 1, then we prefer f(i) to s(i).
            # Opposite is true if the value is -1.
            model.addLConstr(z[i,r,1] <= p[f[i],s[i],r])
            model.addLConstr(z[i,r,-1] <= p[s[i],f[i],r])

    for r in K:
        for l in range(num_cands):
            for m in range(num_cands):
                if l != m:
                    # Constraint (21): If m is not on the ballot, we either prefer l to m or l is not on the ballot.
                    model.addLConstr(n[m,r] <= p[l,m,r] + n[l,r])
                    # Constraint (22): We cannot prefer both l to m and m to l.
                    model.addLConstr(p[l,m,r] + p[m,l,r] <= 1)

    model.setObjective(gp.quicksum([C[i,r] for i in Dims for r in K]), GRB.MINIMIZE)
    model.update()
    return model


def continuous_rest_bordaA(election_data, num_clusters):
    """
    Returns a Gurobi model with centroids restricted to possible ballots using average borda embeddings.
    Notes: The current value set for n candidates is (0...n-1).

    Parameters:
    election_data: dict
        A dictionary matching ballots to weights.
    num_clusters: int   
    """
    all_ballots = list(election_data.keys())
    num_ballots = len(all_ballots)
    num_cands = len(set([item for ranking in all_ballots for item in ranking]))

    ballots = np.array([Borda_vector(ballot, num_cands = num_cands, borda_style='avg', start=1) for ballot in all_ballots])
    weights = np.array([election_data[ballot] for ballot in all_ballots])
    value_set = np.array([l/2 for l in range(1, 2*num_cands + 1)])

    
    num_ballots, n = ballots.shape
    assert weights.shape[0] == num_ballots, "Weights and ballots dimensions do not match."
    D = range(num_ballots)
    K = range(num_clusters)
    V = value_set
    Dims = range(n)
    L = range(1, n + 1)
    model = gp.Model("continuous_rest_k_medians")

    # x[j,r]: 1 if ballot j is assigned to cluster r
    x = {(j,r): model.addVar(vtype=GRB.BINARY, name=f"x[{j},{r}]") 
         for j in D for r in K}

    # z[i,r,v]: 1 if ith dimension of the center of the rth cluster is v
    z = {(i,r,v): model.addVar(vtype=GRB.BINARY, name=f"z[{i},{r},{v}]") 
        for i in Dims for r in K for v in V}

    # z_bar[r, l] (+1 so its 1 to 7)
    z_bar = {(r,l): model.addVar(vtype=GRB.BINARY, name=f"z[{r},{l}]") 
        for r in K for l in L}

    # W[i,r,v]: total weight of points in D assigned to cluster r with coordinate i of value v
    W = {(i,r,v): model.addVar(vtype=GRB.INTEGER, name=f"W[{i},{r},{v}]")
         for i in Dims for r in K for v in V}
                
    # C[i,r]: contribution of coordinate i to cost of cluster r
    # Has to be continuous since some values for borda average are not integral
    C = {(i,r): model.addVar(vtype=GRB.CONTINUOUS, name=f"C[{i},{r}]")
        for i in Dims for r in K}
    
    # Define "big-M" ( is this good enough?)
    M = np.sum(weights) * np.max(value_set)
    
    for j in D:
        # Constraint: Each ballot j is assigned to exactly one cluster r
        model.addLConstr(gp.quicksum([x[j,r] for r in K]) == 1)

    # break symmetry to have cluster r-1 have fewer distinct ballots than cluster r
    for r in range(num_clusters-1):
        model.addLConstr(sum(x[i,r] for i in range(num_ballots))<= sum(x[i,r+1] for i in range(num_ballots)))

    for i in Dims:
        for r in K:
            for v in value_set:
                model.addLConstr(gp.quicksum([weights[j]*x[j,r] for j in D if ballots[j, i] == v]) - W[i,r,v] == 0)
                
    
    for i in Dims:
        for r in K:
            # Constraint: There is exactly one median value for each cluster and coordinate
            model.addLConstr(gp.quicksum(z[i,r,v] for v in V) == 1)            
            for v_target in V:
                # # Constraint: Median finding constraints
                # model.addLConstr(gp.quicksum([W[i,r,v] if v > v_target else -1*W[i,r,v] for v in V]) - M*(1-z[i,r,v_target]) <= 0)
                # model.addLConstr(gp.quicksum([W[i,r,v] if v < v_target else -1*W[i,r,v] for v in V]) - M*(1-z[i,r,v_target]) <= 0)

                # # Constraint: Bound C
                # model.addLConstr(C[i,r] - gp.quicksum([np.abs(v - v_target)*W[i,r,v] for v in V]) <= 0)
                # model.addLConstr(gp.quicksum([np.abs(v - v_target)*W[i,r,v] for v in V]) - C[i,r] - M*(1-z[i,r,v_target]) <= 0)

                # Constraint: Bound C
                model.addLConstr(C[i,r] - gp.quicksum([np.abs(v - v_target)*W[i,r,v] for v in V]) - M*(1-z[i,r,v_target]) <= 0)
                model.addLConstr(gp.quicksum([np.abs(v - v_target)*W[i,r,v] for v in V]) - C[i,r] - M*(1-z[i,r,v_target]) <= 0)

    # One threshold for each cluster r
    for r in K:
        model.addLConstr(gp.quicksum([z_bar[r,l] for l in L]) == 1)

    # Each coordinate must have exactly one value assigned
    for i in Dims:
        for r in K:
            model.addLConstr(gp.quicksum(z[i,r,v] for v in V) == 1)

    for i in Dims:
        for r in K:
            for s in range(int(np.ceil(n/2)), n):
                v = (2*s + 1) / 2
                model.addLConstr(z[i,r,v] == 0)

    for v in V:
        for r in K:
            if v > (n / 2):
                if v in L:
                    model.addLConstr(gp.quicksum([z[i,r,v] for i in Dims]) - gp.quicksum([z_bar[r,l] for l in range(1, int(np.floor(v)) + 1)]) == 0)
                    # model.addLConstr(gp.quicksum([z[i,r,v] for i in Dims]) - gp.quicksum([z_bar[r,l] for l in range(1, int(np.floor(v)) + 1)]) == 0)
            else:
                if v in L:
                    model.addLConstr(gp.quicksum([z[i,r,v] for i in Dims]) - gp.quicksum([z_bar[r,l] for l in range(1, int(np.floor(v)) + 1)]) - (2*v - 1)*z_bar[r, 2*v] == 0)
                else:
                    model.addLConstr(gp.quicksum([z[i,r,v] for i in Dims]) - (2*v - 1)*z_bar[r, 2*v] == 0)
    model.setObjective(gp.quicksum([C[i,r] for i in Dims for r in K]), GRB.MINIMIZE)
    model.update()
    return model


def continuous_rest_bordaP(election_data, num_clusters):
    """
    Returns a Gurobi model with centroids restricted to possible ballots using pessimistic borda embeddings.
    Notes: The current value set for n candidates is (0...n-1).

    Parameters:
    election_data: dict
        A dictionary matching ballots to weights.
    num_clusters: int
    """
    all_ballots = list(election_data.keys())
    num_ballots = len(all_ballots)
    num_cands = len(set([item for ranking in all_ballots for item in ranking]))

    ballots = np.array([Borda_vector(ballot, num_cands = num_cands, borda_style='pes') for ballot in all_ballots])
    weights = np.array([election_data[ballot] for ballot in all_ballots])
    value_set = np.array(range(num_cands))

    num_ballots, num_cand_choose_2 = ballots.shape
    assert weights.shape[0] == num_ballots, "Weights and ballots dimensions do not match."
    D = range(num_ballots)
    K = range(num_clusters)
    V = value_set
    Dims = range(num_cand_choose_2)
    model = gp.Model("continuous_rest_k_medians")

    # x[j,r]: 1 if ballot j is assigned to cluster r
    x = {(j,r): model.addVar(vtype=GRB.BINARY, name=f"x[{j},{r}]") 
         for j in D for r in K}
    
    # z[i,r,v]: 1 if ith dimension of the median of the rth cluster is v
    z = {(i,r,v): model.addVar(vtype=GRB.BINARY, name=f"z[{i},{r},{v}]") 
        for i in Dims for r in K for v in V}

    # W[i,r,v]: total weight of points in D assigned to cluster r with coordinate i of value v
    W = {(i,r,v): model.addVar(vtype=GRB.INTEGER, name=f"W[{i},{r},{v}]")
         for i in Dims for r in K for v in V}
                
    # C[i,r]: contribution of coordinate i to cost of cluster r
    # Has to be continuous since some values for borda are not integral
    C = {(i,r): model.addVar(vtype=GRB.CONTINUOUS, name=f"C[{i},{r}]")
        for i in Dims for r in K}
    
    # Define "big-M" ( is this good enough?)
    M = np.sum(weights) * np.max(value_set)
    
    for j in D:
        # Constraint: Each ballot j is assigned to exactly one cluster r
        model.addLConstr(gp.quicksum([x[j,r] for r in K]) == 1)

    # break symmetry to have cluster r-1 have fewer distinct ballots than cluster r
    for r in range(num_clusters-1):
        model.addLConstr(sum(x[i,r] for i in range(num_ballots))<= sum(x[i,r+1] for i in range(num_ballots)))

    for i in Dims:
        for r in K:
            for v in value_set:
                model.addLConstr(gp.quicksum([weights[j]*x[j,r] for j in D if ballots[j, i] == v]) - W[i,r,v] == 0)
                
    
    for i in Dims:
        for r in K:
            # Constraint: There is exactly one median value for each cluster and coordinate
            model.addLConstr(gp.quicksum(z[i,r,v] for v in V) == 1)            
            for v_target in V:
                # # Constraint: Median finding constraints
                # model.addLConstr(gp.quicksum([W[i,r,v] if v > v_target else -1*W[i,r,v] for v in V]) - M*(1-z[i,r,v_target]) <= 0)
                # model.addLConstr(gp.quicksum([W[i,r,v] if v < v_target else -1*W[i,r,v] for v in V]) - M*(1-z[i,r,v_target]) <= 0)

                # # Constraint: Bound C
                # model.addLConstr(C[i,r] - gp.quicksum([np.abs(v - v_target)*W[i,r,v] for v in V]) <= 0)
                # model.addLConstr(gp.quicksum([np.abs(v - v_target)*W[i,r,v] for v in V]) - C[i,r] - M*(1-z[i,r,v_target]) <= 0)
                # # Constraint: Bound C
                model.addLConstr(C[i,r] - gp.quicksum([np.abs(v - v_target)*W[i,r,v] for v in V]) - M*(1-z[i,r,v_target]) <= 0)
                model.addLConstr(gp.quicksum([np.abs(v - v_target)*W[i,r,v] for v in V]) - C[i,r] - M*(1-z[i,r,v_target]) <= 0)

    # True ballot constraints

    # Each coordinate for each centroid has exactly one value
    for i in Dims:
        for r in K:
            model.addLConstr(gp.quicksum([z[i,r,v] for v in V]) == 1)

    # For each centroid, each value other than 0 appears at most once
    for r in K:
        for v in V[1:]: # Asssuming V is ordered
            model.addLConstr(gp.quicksum([z[i,r,v] for i in Dims]) <= 1)

    # If one of the coordinates is v, then also one of the coordinates must be v+1
    for r in K:
        for v in V[1:-1]:
            model.addLConstr(gp.quicksum([z[i,r,v] for i in Dims]) - gp.quicksum([z[i,r,v+1] for i in Dims]) <= 0)

    model.setObjective(gp.quicksum([C[i,r] for i in Dims for r in K]), GRB.MINIMIZE)

    return model


########################################
# DISCRETE
# MODELS
########################################

def formulate_new(possible_pairs, multiplicities,distances_dict, NUM_CLUSTERS, NUM_OUTLIERS=0):
    """
    Generates a Gurobi model for the clustering problem.

    Parameters:
    possible_pairs: dict
        A dictionary of possible pairs of points (ballots).
    multiplicities: dict
        A dictionary matching each ballot to its count.
    distances_dict: dict
        A dictionary matching each pair of ballots to their distance.
    NUM_CLUSTERS: int
        The number of clusters to be formed.
    NUM_OUTLIERS: int
        The number of outliers to be allowed.
    """
    m = gp.Model("clustering_new")
    m.Params.MIPGapAbs = 0.1 
    pairs = {}
    isCenter = {}
    outlier = {}
    for i in possible_pairs.keys():
        isCenter[i] = m.addVar(vtype=GRB.BINARY, name = "isCenter[%s]" %i)
        outlier[i] = m.addVar(vtype=GRB.BINARY, name = "outlier[%s]" %i)
        for j in possible_pairs[i]:
            pairs[i,j] = m.addVar(vtype=GRB.BINARY, name = "pair{%s,%s}" % (i,j))

    for j in possible_pairs.keys():#constraint to define the isNotCenter variable
        m.addConstr(isCenter[j] <= sum(pairs[i,j] for i in possible_pairs[j])) #isCenter[j] is LEQ than sum of all pairs [i,j]
        for i in possible_pairs[j]:#to ensure that isCenter is 1 if any point has it as a center
            m.addConstr(isCenter[j] - pairs[i,j] >= 0)

    for i in possible_pairs.keys():
        # this constraint was changed to ensure that each point i is either part of at least one cluster, or declared an outlier
        m.addConstr(sum(pairs[i,j] for j in possible_pairs[i]) + outlier[i] >= 1) 
        m.addConstr(pairs[i,i] >= isCenter[i])

    m.addConstr(sum(isCenter[j] for j in possible_pairs.keys()) == NUM_CLUSTERS)
    m.addConstr(sum(multiplicities[i]*outlier[i] for i in possible_pairs.keys()) <= NUM_OUTLIERS)

    m.setObjective(sum(multiplicities[i]*distances_dict[i,j]*pairs[i,j] for i in possible_pairs.keys() for j in possible_pairs[i]), GRB.MINIMIZE)
    return m, pairs, isCenter, outlier

def discrete_HH(election, num_clusters):
    """
    Returns a Gurobi model for the discrete k-medians problem (clustering) using head-to-head embeddings.

    Parameters:
    election: dict
        A dictionary matching ballots to weights.
    num_clusters: int
        The number of clusters to be formed.
    """
    all_ballots = list(election.keys())
    num_ballots = len(all_ballots)
    candidates = sorted(list(set([item for ranking in all_ballots for item in ranking])))
    num_cands = len(candidates)
    sample_weight = np.array([election[ballot] for ballot in all_ballots])
    
    TOLERANCE = 14

    distances_dict = {}
    possible_pairs = {}
    
        
    for i in range(len(all_ballots)):
        temp_list = []
        for j in range(len(all_ballots)):
            distances_dict[(i,j)] = HH_dist(all_ballots[i],all_ballots[j],num_cands, order = 1)
            if True: #(distances_dict[(i,j)] <= TOLERANCE): # Not sure what tolerance should be, so ignoring. If it is too slow, consider adding some tolerance.
                temp_list.append(j)
        possible_pairs[i] = temp_list

    m, pairs, isCenter, outlier = formulate_new(possible_pairs, sample_weight, distances_dict, NUM_CLUSTERS = num_clusters)
    return m
    

def discrete_bordaP(election, num_clusters):
    """
    Returns a Gurobi model for the discrete k-medians problem (clustering) using pessimistic borda embeddings.
    Note: The current value set for n candidates is (0...n-1).

    Parameters:
    election: dict
        A dictionary matching ballots to weights.
    num_clusters: int
        The number of clusters to be formed.
    """
    all_ballots = list(election.keys())
    num_ballots = len(all_ballots)
    candidates = sorted(list(set([item for ranking in all_ballots for item in ranking])))
    num_cands = len(candidates)
    sample_weight = np.array([election[ballot] for ballot in all_ballots])
    
    TOLERANCE = 14
    
    # preprocess data again
    distances_dict = {}
    possible_pairs = {}
    
        
    for i in range(len(all_ballots)):
        temp_list = []
        for j in range(len(all_ballots)):
            distances_dict[(i,j)] = Borda_dist(all_ballots[i],all_ballots[j],num_cands, borda_style='pes', order = 1)
            if True:#(distances_dict[(i,j)] <= TOLERANCE): # Not sure what tolerance should be, so ignoring. If it is too slow, consider adding some tolerance.
                temp_list.append(j)
        possible_pairs[i] = temp_list
    m, pairs, isCenter, outlier = formulate_new(possible_pairs, sample_weight, distances_dict, NUM_CLUSTERS = 2)
    return m

def discrete_bordaA(election, num_clusters):
    """
    Returns a Gurobi model for the discrete k-medians problem (clustering) using average borda embeddings.
    Note: The current value set for n candidates is (0...n-1).

    Parameters:
    election: dict
        A dictionary matching ballots to weights.
    num_clusters: int
        The number of clusters to be formed.
    """
    all_ballots = list(election.keys())
    num_ballots = len(all_ballots)
    candidates = sorted(list(set([item for ranking in all_ballots for item in ranking])))
    num_cands = len(candidates)
    sample_weight = np.array([election[ballot] for ballot in all_ballots])
    
    
    # preprocess data again
    distances_dict = {}
    possible_pairs = {}
    
        
    for i in range(len(all_ballots)):
        temp_list = []
        for j in range(len(all_ballots)):
            distances_dict[(i,j)] = Borda_dist(all_ballots[i],all_ballots[j],num_cands, borda_style='avg', order = 1)
            if True:#(distances_dict[(i,j)] <= TOLERANCE): # Not sure what tolerance should be, so ignoring. If it is too slow, consider adding some tolerance.
                temp_list.append(j)
        possible_pairs[i] = temp_list

    m, pairs, isCenter, outlier = formulate_new(possible_pairs, sample_weight, distances_dict, NUM_CLUSTERS = 2)
    return m