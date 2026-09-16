import numpy as np
import networkx as nx

# =====================================================================
# 1. GLOBAL CONSTANTS (The Structural Laws of Physics)
# =====================================================================
NUM_NODES = 30
DT = 0.05
SIGMA = 0.5         # Michaelis-Menten saturation bottleneck
GAMMA = 2.0         # Mechanical contraction velocity coefficient
K_BASELINE = 1.0    # Structural baseline spring stiffness
TAU = 5             # Trophic memory expectation window (in clock steps)
ETA = 0.01          # Homeostatic gate decay rate (tending back to baseline)
VARIANCE_WINDOW = 20 # Number of steps to track local boredom

# Mass-action chemical decay rates
DECAY_RATES = np.array([0.1, 0.1, 0.1, 0.1]) # [d_a, d_b, d_c, d_d]

# Constant Stoichiometric Matrix (A -> B -> C -> D loop mechanics)
STOICH_MATRIX = np.array([
    [-1.0,  0.0,  0.0,  0.0],  # A consumption
    [ 1.0,  0.0,  0.0, -1.0],  # B production / inhibition destruction
    [ 0.0,  1.0, -1.0,  0.0],  # C production via B / conversion to D
    [ 0.0,  0.0,  1.0,  0.0]   # D production via C
])

# Reaction rate scale constants (k1, k2, k3, k4)
K_RATES = np.array([0.2, 1.5, 0.4, 2.0])

# =====================================================================
# 2. ALIFE MESH ARCHITECTURE & NODE PHENOTYPES
# =====================================================================
nodes = list(range(NUM_NODES))
type_assignments = ['Era']*6 + ['Con']*12 + ['Dup']*12
np.random.shuffle(type_assignments)

adj_matrix = np.zeros((NUM_NODES, NUM_NODES))

# Connect Era leaf nodes downstream
for i in range(NUM_NODES):
    if type_assignments[i] == 'Era':
        target = np.random.choice([j for j in range(NUM_NODES) if type_assignments[j] != 'Era'])
        adj_matrix[i, target] = 1.0

# Connect Con and Dup nodes randomly (targeting degree <= 3)
triangles = [i for i in range(NUM_NODES) if type_assignments[i] in ['Con', 'Dup']]
for i in triangles:
    available_targets = [j for j in range(NUM_NODES) if i != j and adj_matrix[i, j] == 0]
    if len(available_targets) > 0:
        num_connections = np.random.choice([1, 2, 3])
        targets = np.random.choice(available_targets, size=min(num_connections, len(available_targets)), replace=False)
        for t in targets:
            adj_matrix[i, t] = 1.0

# =====================================================================
# 3. PER-AGENT PARAMETERS & STATE INITIALIZATION
# =====================================================================
chemical_states = np.random.uniform(0.1, 0.5, size=(NUM_NODES, 4))
positions = np.random.uniform(-10.0, 10.0, size=(NUM_NODES, 2))

rest_lengths = np.zeros((NUM_NODES, NUM_NODES))
for i in range(NUM_NODES):
    for j in range(NUM_NODES):
        if adj_matrix[i, j] == 1.0 or adj_matrix[j, i] == 1.0:
            rest_lengths[i, j] = np.linalg.norm(positions[i] - positions[j])

ambient_feeds = np.zeros((NUM_NODES, 4))
for i in range(NUM_NODES):
    if type_assignments[i] != 'Era':
        ambient_feeds[i, 0] = 0.05  # Slow background trickle of Fuel A

threshold_gates = np.ones((NUM_NODES, 4)) * 0.4  # Core learnable parameters
transmission_rates = np.zeros(NUM_NODES)

# Meta-Learning Parameters (Boredom Mechanics)
learning_rates = np.zeros(NUM_NODES)             # Dynamic mu_i
baseline_learning_rates = np.zeros(NUM_NODES)

for i in range(NUM_NODES):
    if type_assignments[i] == 'Con':
        transmission_rates[i] = 0.8
        baseline_learning_rates[i] = 0.05
    elif type_assignments[i] == 'Dup':
        transmission_rates[i] = 0.6
        baseline_learning_rates[i] = 0.03
    learning_rates[i] = baseline_learning_rates[i]

# Historical tracking metrics
fuel_history = np.zeros((TAU, NUM_NODES))
signal_history = np.zeros((VARIANCE_WINDOW, NUM_NODES)) # Tracks Catalyst C for boredom evaluation

food_source = np.array([2.0, 2.0])

# =====================================================================
# 4. THE REACTION-DIFFUSION-LEARNING LOOP WITH BOREDOM
# =====================================================================
def step_simulation(step_count):
    global chemical_states, threshold_gates, positions, fuel_history, signal_history, learning_rates
    
    # 4.1 Environmental Ingestion
    for i in range(NUM_NODES):
        if type_assignments[i] == 'Era':
            distance_to_food = np.linalg.norm(positions[i] - food_source)
            if distance_to_food < 1.5:
                chemical_states[i, 0] += 0.8 * DT
                chemical_states[i, 1] += 0.2 * DT

    # 4.2 Local Reactor Vats
    next_chemical_states = np.copy(chemical_states)
    for i in range(NUM_NODES):
        A, B, C, D = chemical_states[i]
        
        r1 = K_RATES[0] * A
        r2 = (K_RATES[1] * B * C) / (1.0 + SIGMA * C)
        r3 = K_RATES[2] * C
        r4 = K_RATES[3] * B * D
        rates = np.array([r1, r2, r3, r4])
        
        decay = DECAY_RATES * chemical_states[i]
        stoich_change = STOICH_MATRIX.T @ rates
        next_chemical_states[i] += (-decay + ambient_feeds[i] + stoich_change) * DT

    # 4.3 Asymmetric Network Signaling
    net_network_flux = np.zeros((NUM_NODES, 4))
    for j in range(NUM_NODES):
        for i in range(NUM_NODES):
            if adj_matrix[j, i] == 1.0:
                if type_assignments[j] == 'Con' and chemical_states[j, 2] > threshold_gates[j, 2]:
                    net_network_flux[i, 2] += transmission_rates[j] * (chemical_states[j, 2] - threshold_gates[j, 2]) * DT
                elif type_assignments[j] == 'Dup' and chemical_states[j, 3] > threshold_gates[j, 3]:
                    net_network_flux[i, 3] += transmission_rates[j] * (chemical_states[j, 3] - threshold_gates[j, 3]) * DT
                elif type_assignments[j] == 'Era':
                    net_network_flux[i, 0] += 0.5 * chemical_states[j, 0] * DT
                    net_network_flux[i, 1] += 0.5 * chemical_states[j, 1] * DT

    chemical_states = np.clip(next_chemical_states + net_network_flux, 0.0, 5.0)

    # 4.4 Phase-Coupled Actuation Engine
    forces = np.zeros((NUM_NODES, 2))
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            if adj_matrix[i, j] == 1.0 or adj_matrix[j, i] == 1.0:
                vec = positions[j] - positions[i]
                dist = np.linalg.norm(vec)
                if dist > 0:
                    k_active = K_BASELINE * (1.0 + GAMMA * max(chemical_states[i, 2], chemical_states[j, 2]))
                    force_mag = k_active * (dist - rest_lengths[i, j])
                    forces[i] += force_mag * (vec / dist)
    positions += forces * DT * 0.1

    # 4.5 Trophic Learning & Boredom Modulation (Meta-Learning Engine)
    fuel_index = step_count % TAU
    var_index = step_count % VARIANCE_WINDOW
    
    # Store current Catalyst C levels to monitor dynamic fluctuation
    signal_history[var_index] = chemical_states[:, 2]
    old_fuel_states = fuel_history[fuel_index]
    
    for i in range(NUM_NODES):
        if type_assignments[i] in ['Con', 'Dup']:
            # A. Calculate Boredom (Variance of signaling over time window)
            local_signal_variance = np.var(signal_history[:, i])
            
            # Boredom Threshold: If variance is near zero, hyper-inflate plasticity
            if local_signal_variance < 0.005:
                # Evolutionary panic button: learning rate scales up to force rapid mutation
                learning_rates[i] = min(learning_rates[i] * 1.15, 0.5) 
            else:
                # Homeostatic stabilization back to natural baseline
                learning_rates[i] = learning_rates[i] * 0.95 + baseline_learning_rates[i] * 0.05
            
            # B. Standard Trophic Parameter Update
            delta_fuel = chemical_states[i, 0] - old_fuel_states[i]
            correlation_c = chemical_states[i, 2] * delta_fuel
            
            # Gate update scaling directly with the dynamic learning rate
            threshold_gates[i, 2] += (ETA * (0.4 - threshold_gates[i, 2]) - learning_rates[i] * correlation_c) * DT
            threshold_gates[i, 2] = np.clip(threshold_gates[i, 2], 0.05, 0.95)

    fuel_history[fuel_index] = chemical_states[:, 0]

# =====================================================================
# 5. RUN SIMULATION
# =====================================================================
print("Launching MKA-Framework with Boredom-Based Meta-Learning...")
for loop_step in range(120):
    step_simulation(loop_step)
    if loop_step % 20 == 0:
        avg_mu = np.mean(learning_rates[[i for i in range(NUM_NODES) if type_assignments[i] in ['Con', 'Dup']]])
        avg_gate = np.mean(threshold_gates[:, 2])
        print(f"Step {loop_step:03d} | Avg Plasticity (Mu): {avg_mu:.4f} | Avg Gate C: {avg_gate:.4f}")
