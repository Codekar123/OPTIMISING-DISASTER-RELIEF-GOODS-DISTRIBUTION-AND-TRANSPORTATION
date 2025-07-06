import pulp as lp

# Define the problem
problem = lp.LpProblem("Disaster_Relief_Goods_Routing_Optimization", lp.LpMinimize)

# Parameters
RO = 10          # Total number of relief orders
W = 3            # Total number of warehouses
V = 2            # Total number of vehicles
D = 4            # Total number of disaster regions
C = 3            # Number of cargos per vehicle

# Sets
orders = range(RO)
warehouses = range(W)
vehicles = range(V)
regions = range(D)
cargos = range(C)

# Inputs
# Distances between warehouses and disaster regions (in km)
distance_warehouse_to_region = {
    (0, 0): 20, (0, 1): 35, (0, 2): 50, (0, 3): 25,
    (1, 0): 30, (1, 1): 25, (1, 2): 20, (1, 3): 15,
    (2, 0): 45, (2, 1): 40, (2, 2): 35, (2, 3): 30,
}

# Vehicle average speed (in km/h)
vehicle_speed = 40  # Assume all vehicles travel at the same speed

# Travel time between warehouses and regions (in hours)
travel_time_warehouse_to_region = {
    (w, d, v): distance_warehouse_to_region[w, d] / vehicle_speed
    for w in warehouses for d in regions for v in vehicles
}

# Size of each order (in units)
size = {o: 5 for o in orders}  # Replace with actual order sizes if needed

# Vehicle capacities (in units)
vehicle_capacity = {v: 30 for v in vehicles}

# Ready times for each order at each warehouse (in hours)
ready_time = {(w, o): 1 for w in warehouses for o in orders}

# Order destinations (determine which region each order is meant for)
destination = {(d, o): 1 if o % D == d else 0 for d in regions for o in orders}

# Allowable warehouses for orders (all are valid in this example)
allowable_warehouse = {(o, w): 1 for o in orders for w in warehouses}

# Allowable vehicles for orders (all are valid in this example)
conflict_vehicle_order = {(o, v): 1 for o in orders for v in vehicles}

# Big-M constant for constraints
M = 1000

# Decision Variables
delivery_time = lp.LpVariable.dicts("DeliveryTime", orders, lowBound=0, cat="Continuous")
loading_time = lp.LpVariable.dicts("LoadingTime", orders, lowBound=0, cat="Continuous")
warehouse_assignment = lp.LpVariable.dicts("WarehouseAssignment", [(o, w) for o in orders for w in warehouses], cat="Binary")
vehicle_assignment = lp.LpVariable.dicts("VehicleAssignment", [(v, c, o) for v in vehicles for c in cargos for o in orders], cat="Binary")
route_assignment = lp.LpVariable.dicts("RouteAssignment", [(w, d, v, o) for w in warehouses for d in regions for v in vehicles for o in orders], cat="Binary")

# Objective: Minimize total delivery time
problem += lp.lpSum(delivery_time[o] for o in orders), "Minimize_Total_Delivery_Time"

# Constraints

# 1. Each order must be assigned to only one warehouse
for o in orders:
    problem += lp.lpSum(warehouse_assignment[o, w] for w in warehouses) == 1

# 2. Each order must be assigned to only one vehicle and cargo
for o in orders:
    problem += lp.lpSum(vehicle_assignment[v, c, o] for v in vehicles for c in cargos) == 1

# 3. Each order must have a single route from a warehouse to a region
for o in orders:
    problem += lp.lpSum(route_assignment[w, d, v, o] for w in warehouses for d in regions for v in vehicles) == 1

# 4. Link route assignment with warehouse assignment
for o in orders:
    for w in warehouses:
        for d in regions:
            for v in vehicles:
                problem += route_assignment[w, d, v, o] <= warehouse_assignment[o, w]

# 5. Link route assignment with destination
for o in orders:
    for w in warehouses:
        for d in regions:
            for v in vehicles:
                if destination[d, o] == 0:
                    problem += route_assignment[w, d, v, o] == 0

# 6. Vehicle capacity constraints
for v in vehicles:
    for c in cargos:
        problem += lp.lpSum(size[o] * vehicle_assignment[v, c, o] for o in orders) <= vehicle_capacity[v]

# 7. Loading time constraint based on warehouse ready time
for o in orders:
    for w in warehouses:
        problem += loading_time[o] >= ready_time[w, o] - M * (1 - warehouse_assignment[o, w])

# 8. Delivery time constraint based on route travel time
for o in orders:
    for w in warehouses:
        for d in regions:
            for v in vehicles:
                if destination[d, o] == 1:
                    problem += (
                        delivery_time[o] >= loading_time[o] + travel_time_warehouse_to_region[w, d, v]
                        - M * (1 - route_assignment[w, d, v, o])
                    )

# Solve the problem
problem.solve()

# Output the results
print("Status:", lp.LpStatus[problem.status])
print("Total Delivery Time:", lp.value(problem.objective))
for o in orders:
    print(f"Delivery time for order {o}: {lp.value(delivery_time[o])}")
for o in orders:
    for w in warehouses:
        if lp.value(warehouse_assignment[o, w]) == 1:
            print(f"Order {o} assigned to warehouse {w}")
for w in warehouses:
    for d in regions:
        for v in vehicles:
            for o in orders:
                if lp.value(route_assignment[w, d, v, o]) == 1:
                    print(f"Order {o} assigned to route: Warehouse {w} -> Region {d} using Vehicle {v}")

