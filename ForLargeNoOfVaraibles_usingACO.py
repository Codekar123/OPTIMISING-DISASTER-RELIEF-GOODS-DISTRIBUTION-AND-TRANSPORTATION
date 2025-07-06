import random
import numpy as np

# Parameters
RO = 100         # Total number of relief orders
W = 15           # Total number of warehouses
V = 10           # Total number of vehicles
D = 4            # Total number of disaster regions
C = 5            # Number of cargos per vehicle

# Inputs
# Predefined distances (in km) between warehouses and disaster regions
distance_warehouse_to_region = {
    (0, 0): 25, (0, 1): 40, (0, 2): 55, (0, 3): 35,
    (1, 0): 30, (1, 1): 50, (1, 2): 45, (1, 3): 20,
    (2, 0): 50, (2, 1): 65, (2, 2): 60, (2, 3): 40,
    (3, 0): 20, (3, 1): 35, (3, 2): 50, (3, 3): 25,
    (4, 0): 60, (4, 1): 55, (4, 2): 70, (4, 3): 50,
    (5, 0): 30, (5, 1): 20, (5, 2): 40, (5, 3): 30,
    (6, 0): 45, (6, 1): 40, (6, 2): 65, (6, 3): 45,
    (7, 0): 35, (7, 1): 25, (7, 2): 40, (7, 3): 30,
    (8, 0): 55, (8, 1): 50, (8, 2): 60, (8, 3): 45,
    (9, 0): 25, (9, 1): 20, (9, 2): 35, (9, 3): 20,
    (10, 0): 60, (10, 1): 50, (10, 2): 70, (10, 3): 55,
    (11, 0): 45, (11, 1): 40, (11, 2): 50, (11, 3): 35,
    (12, 0): 35, (12, 1): 30, (12, 2): 40, (12, 3): 25,
    (13, 0): 50, (13, 1): 60, (13, 2): 70, (13, 3): 55,
    (14, 0): 20, (14, 1): 25, (14, 2): 35, (14, 3): 30,
}

# Vehicle average speed (in km/h)
vehicle_speed = 40  # Assume all vehicles travel at the same speed

# Travel time between warehouses and regions (in hours)
travel_time_warehouse_to_region = {
    (w, d): distance / vehicle_speed
    for (w, d), distance in distance_warehouse_to_region.items()
}

# Order sizes (all orders have a size of 5 units)
order_sizes = {o: 5 for o in range(RO)}

# Vehicle capacities (20 units each)
vehicle_capacity = 50

# Destination assignments (each order is pre-assigned to a region)
order_destinations = {o: o % D for o in range(RO)}

# ACO Parameters
num_ants = 30
num_iterations = 200
alpha = 1.0
beta = 2.0
rho = 0.1
Q = 1.0

# Initialize pheromone levels
pheromone = {(w, d): 1.0 for w in range(W) for d in range(D)}

# Heuristic: Inverse of travel time
heuristic = {key: 1 / value for key, value in travel_time_warehouse_to_region.items()}

# ACO Algorithm
def ant_colony_optimization():
    global pheromone

    best_solution = None
    best_cost = float("inf")
    best_vehicle_assignments = None

    for iteration in range(num_iterations):
        solutions = []
        costs = []
        vehicle_assignments = []

        for ant in range(num_ants):
            # Construct a solution
            solution = []
            total_cost = 0
            # Track vehicle usage separately for each ant (reset it here)
            vehicles_used = {v: 0 for v in range(V)}  # Track vehicle capacities for this ant
            vehicle_assignment = []

            for o in range(RO):
                # Select warehouse and region for the order
                w_d_choices = [
                    (w, d)
                    for w in range(W)
                    for d in range(D)
                    if order_destinations[o] == d
                ]

                probabilities = [
                    (pheromone[w, d] ** alpha) * (heuristic[w, d] ** beta)
                    for w, d in w_d_choices
                ]
                probabilities_sum = sum(probabilities)
                probabilities = [p / probabilities_sum for p in probabilities]

                # Roulette wheel selection
                choice = random.choices(w_d_choices, weights=probabilities, k=1)[0]
                w, d = choice

                # Assign a vehicle (resetting vehicle usage check for each ant)
                assigned_vehicle = None
                for v in range(V):
                    if vehicles_used[v] + order_sizes[o] <= vehicle_capacity:
                        vehicles_used[v] += order_sizes[o]  # Update the capacity used
                        assigned_vehicle = v
                        break

                if assigned_vehicle is None:
                    raise Exception(f"No feasible vehicle assignment for order {o}.")

                # Record assignment
                solution.append((o, w, d, assigned_vehicle))
                vehicle_assignment.append((o, assigned_vehicle))
                total_cost += travel_time_warehouse_to_region[w, d]

            solutions.append(solution)
            costs.append(total_cost)
            vehicle_assignments.append(vehicle_assignment)

            # Update best solution
            if total_cost < best_cost:
                best_solution = solution
                best_cost = total_cost
                best_vehicle_assignments = vehicle_assignment

        # Update pheromone levels
        pheromone_update(solutions, costs)

        print(f"Iteration {iteration + 1}/{num_iterations}: Best Cost = {best_cost}")

    return best_solution, best_cost, best_vehicle_assignments

def pheromone_update(solutions, costs):
    global pheromone

    # Evaporate pheromone
    for key in pheromone:
        pheromone[key] *= (1 - rho)

    # Deposit pheromone
    for solution, cost in zip(solutions, costs):
        for o, w, d, _ in solution:
            pheromone[w, d] += Q / cost

# Solve the problem using ACO
best_solution, best_cost, best_vehicle_assignments = ant_colony_optimization()

# Output results
print("\nBest Solution:")
for o, w, d, v in best_solution:
    print(f"Order {o} -> Warehouse {w} -> Region {d} using Vehicle {v}")
print(f"Total Delivery Time: {best_cost:.2f} hours")
print("\nVehicle Assignments:")
for o, v in best_vehicle_assignments:
    print(f"Order {o} assigned to Vehicle {v}")


