# Functions for extracting and analyzing model results
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def get_centroids(sol_file: str, is_discrete: bool, proxy: str, ballots: np.ndarray = None) -> np.ndarray:
    """
    Get the centroid of the model results

    Args:
    sol_file: str
        The path to the solution file
    is_discrete: bool
        Whether the model is discrete or not
    proxy: str
        The proxy for the model. If proxy is "bordaa", subtracts 1 from each dimension since
        bordaA in the model starts at 1.
    ballots: np.ndarray
        The list of ballots

    Returns:
    centroids: dict
        Dictionary where centroids[c] is the numpy array for centroid c
    """
    with open(sol_file, 'r') as file:
        # For discrete models, there is a variable isCenter that indicates the index of the centroid ballot
        # Make sure the order of ballots is consistent with the order in the model
        if is_discrete:
            if ballots is None:
                raise ValueError("Ballots must be provided for discrete models")
            
            lines = file.readlines()
            centroids = {}
            c_index = 0
            for line in lines:
                if line.startswith('isCenter['):
                    index = int(line[line.find('[')+1:line.find(']')])
                    value = float(line.split()[-1])
                    
                    if value == 1:
                        centroids[c_index] = ballots[index]
                        c_index += 1
            
            return centroids
        else:
            lines = file.readlines()
            centroid_dict = {}
            for line in lines:
                if line.startswith('z['):
                    bracket_content = line[line.find('[')+1:line.find(']')]
                    parts = bracket_content.split(',')
                    if len(parts) != 3:
                        # Needed since continuous_rest bordaA has another z variable
                        continue
                    dim = int(parts[0])
                    c = int(parts[1])
                    v = float(parts[2])
                    value = int(line.split()[-1])
                    
                    if value == 1:
                        if c not in centroid_dict:
                            centroid_dict[c] = {}
                        centroid_dict[c][dim] = v

            # Create dictionary of centroids
            centroids = {}
            for c in centroid_dict:
                n_dims = max(centroid_dict[c].keys()) + 1
                centroid_array = np.zeros(n_dims)
                for dim in centroid_dict[c]:
                    # Adjust by -1 for bordaA
                    centroid_array[dim] = centroid_dict[c][dim] - 1 if proxy == 'bordaa' else centroid_dict[c][dim]
                centroids[c] = tuple(centroid_array)
                
            return centroids
        

def check_continuous_solution(model, centroids, proxy, election_name='election', verbose=True):
    # Correctly copy the model to avoid modifying the original
    model_copy = model.copy()
    model_copy.Params.LogToConsole = 1 if verbose else 0

    var_dict = {var.VarName: var for var in model_copy.getVars()}
    # Set z variables to 1
    for c_index, centroid in centroids.items():
        centroid = np.array(centroid)
        for dim in range(len(centroid)):
            if proxy == 'bordaa':
                value = float(centroid[dim])
            else:
                value = int(centroid[dim])
            var_name = f'z[{dim},{c_index},{value}]'
            if var_name not in var_dict:
                raise KeyError(f"Variable {var_name} not found in model variables.")
            var = var_dict[var_name]
            var.LB = 1
            var.UB = 1
    
    model_copy.update()
    model_copy.optimize()

    if model_copy.Status != GRB.OPTIMAL:
        print("Solution is infeasible")
        
        # Compute IIS to identify conflicting constraints and bounds
        model_copy.computeIIS()
        model_copy.write(f"{election_name}.ilp")  # Save IIS to a file
        
        # Collect violated constraints and bounds
        violated_constraints = [constr.ConstrName for constr in model_copy.getConstrs() if constr.IISConstr]
        violated_bounds = []
        for var in model_copy.getVars():
            if var.IISLB:
                violated_bounds.append(f"{var.VarName} (Lower Bound)")
            if var.IISUB:
                violated_bounds.append(f"{var.VarName} (Upper Bound)")
        
        print("\nViolated Constraints:")
        for constr in violated_constraints:
            print(f" - {constr}")
        
        print("\nViolated Variable Bounds:")
        for bound in violated_bounds:
            print(f" - {bound}")
        
        print("\nDetailed IIS written to 'model_iis.ilp'")
        return False
    else:
        return True

# def check_continuous_solution(model, centroids, proxy, verbose=True):
#     # Copy model
#     model_copy = model
#     model_copy.Params.LogToConsole = 1 if verbose else 0

#     var_dict = {var.VarName: var for var in model_copy.getVars()}
#     # Set z variables to 1
#     for c_index, centroid in centroids.items():
#         centroid = np.array(centroid)
#         for dim in range(len(centroid)):
#             if proxy == 'bordaa':
#                 value = centroid[dim] + 1
#             else:
#                 value = int(centroid[dim])
#             var_dict[f'z[{dim},{c_index},{value}]'].LB = 1
#             var_dict[f'z[{dim},{c_index},{value}]'].UB = 1
    
#     model_copy.update()

#     model_copy.optimize()

#     if model_copy.Status != GRB.OPTIMAL:
#         print("Solution is infeasible")
#         return False
#     else:
#         return True
        

def extract_centroids(model, z, K, Dims, V):
    import numpy as np
    
    # Initialize centroid array
    centroids = np.zeros((len(K), len(Dims)))
    
    # For each cluster and dimension
    for r in K:
        for i in Dims:
            # Find the value v where z[i,r,v] = 1
            for v in V:
                if abs(z[i,r,v] - 1.0) < 1e-6:  # Check if binary variable is 1 (with tolerance)
                    centroids[r,i] = v
                    break
    
    return centroids



def extract_z_variables(model, Dims, K, V):
    # Initialize the result dictionary
    result = {}
    
    # Create a mapping of variable names to their expected tuples
    name_to_tuple = {f"z[{i},{r},{v}]": (i, r, v) 
                    for i in Dims for r in K for v in V}
    print(name_to_tuple)
    # Iterate through variables and extract binary values
    for var in model.getVars():

        if var.VarName in name_to_tuple:
            indices = name_to_tuple[var.VarName]
            # Round to ensure we get exactly 0 or 1
            # Why .Start? Not sure, for some reason .X doesn't work, maybe try running model.optimize() first
            # Nice thing is when you load solutions from file, the variables start with .start
            result[indices] = round(var.X)
    
    return result


def assign_ballots(centroids: dict, original_ballots: dict, proxy_conversion: dict) -> dict:
    clusters = {i: {} for i in centroids}
    cost = 0
    for i, (original_ballot, count) in enumerate(original_ballots.items()):
       proxy_ballot = tuple(proxy_conversion[original_ballot])
       original_ballot = tuple(original_ballot)
       distances = {c: sum(abs(b - p) for b, p in zip(proxy_ballot, centroids[c])) for c in centroids}
       min_dist = min(distances.values())
       closest = [c for c, d in distances.items() if d == min_dist]
       split_count = count / len(closest)
       cost += min_dist * count
       for c in closest:
           clusters[c][i, original_ballot] = clusters[c].get(original_ballot, 0) + split_count

    # Assert that sum of weights in clusters is same as sum of original weights in both proxy and original
    assert sum(sum(c.values()) for c in clusters.values()) == sum(original_ballots.values())
    return clusters, cost

if __name__ == '__main__':
    sol_file = 'all_models/03_cand_result/continuous_bordaA_eilean_siar_2022_ward3_03_2.sol'
    is_discrete = False
    centroids = get_centroids(sol_file, is_discrete, None)
    print(centroids)
