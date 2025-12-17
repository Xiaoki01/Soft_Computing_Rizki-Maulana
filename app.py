from flask import Flask, render_template, request, jsonify
import random
import numpy as np

app = Flask(__name__)

# Route untuk halaman home
@app.route('/')
def home():
    return render_template('home.html')

# Route untuk halaman tugas 1
@app.route('/tugas1')
def tugas1():
    return render_template('tugas1.html')

# Route untuk halaman tugas 2
@app.route('/tugas2')
def tugas2():
    return render_template('tugas2.html')

# Route untuk halaman tugas 3
@app.route('/tugas3')
def tugas3():
    return render_template('tugas3.html')

# Route untuk halaman tugas 4
@app.route('/tugas4')
def tugas4():
    return render_template('tugas4.html')

@app.route('/calculate_fuzzy', methods=['POST'])
def calculate_fuzzy():
    data = request.get_json()
    temperature = float(data['temperature'])
    humidity = float(data['humidity'])
    
    # Fungsi keanggotaan Temperature
    def temp_cold(t):
        if t <= 20:
            return 1
        elif 20 < t < 40:
            return (40 - t) / 20
        else:
            return 0
    
    def temp_normal(t):
        if t <= 20:
            return 0
        elif 20 < t < 40:
            return (t - 20) / 20
        elif 40 <= t <= 60:
            return 1
        elif 60 < t < 80:
            return (80 - t) / 20
        else:
            return 0
    
    def temp_hot(t):
        if t <= 60:
            return 0
        elif 60 < t < 80:
            return (t - 60) / 20
        else:
            return 1
    
    # Fungsi keanggotaan Humidity
    def hum_low(h):
        if h <= 45:
            return 1
        elif 45 < h < 50:
            return (50 - h) / 5
        else:
            return 0
    
    def hum_normal(h):
        if h <= 45:
            return 0
        elif 45 < h < 50:
            return (h - 45) / 5
        elif 50 <= h <= 55:
            return 1
        elif 55 < h < 60:
            return (60 - h) / 5
        else:
            return 0
    
    def hum_high(h):
        if h <= 55:
            return 0
        elif 55 < h < 60:
            return (h - 55) / 5
        else:
            return 1
    
    # Hitung derajat keanggotaan
    temp_memberships = {
        'cold': temp_cold(temperature),
        'normal': temp_normal(temperature),
        'hot': temp_hot(temperature)
    }
    
    hum_memberships = {
        'low': hum_low(humidity),
        'normal': hum_normal(humidity),
        'high': hum_high(humidity)
    }
    
    # Rules dan konsekuen
    rules = [
        {'temp': 'cold', 'hum': 'low', 'weight': 1, 'result': 0.33},
        {'temp': 'cold', 'hum': 'normal', 'weight': 2, 'result': 0.50},
        {'temp': 'cold', 'hum': 'high', 'weight': 3, 'result': 0.66},
        {'temp': 'normal', 'hum': 'low', 'weight': 1, 'result': 0.50},
        {'temp': 'normal', 'hum': 'normal', 'weight': 2, 'result': 0.66},
        {'temp': 'normal', 'hum': 'high', 'weight': 3, 'result': 0.83},
        {'temp': 'hot', 'hum': 'low', 'weight': 1, 'result': 0.66},
        {'temp': 'hot', 'hum': 'normal', 'weight': 2, 'result': 0.83},
        {'temp': 'hot', 'hum': 'high', 'weight': 3, 'result': 1.00}
    ]
    
    # Hitung alpha dan z untuk setiap rule
    rule_results = []
    numerator = 0
    denominator = 0
    
    for i, rule in enumerate(rules, 1):
        alpha = min(temp_memberships[rule['temp']], hum_memberships[rule['hum']])
        z = rule['result']
        
        if alpha > 0:
            numerator += alpha * z
            denominator += alpha
            
            rule_results.append({
                'rule': f"R{i}",
                'condition': f"Temp {rule['temp'].capitalize()} AND Humidity {rule['hum'].capitalize()}",
                'alpha': round(alpha, 3),
                'z': z
            })
    
    # Defuzzifikasi
    if denominator > 0:
        final_result = numerator / denominator
    else:
        final_result = 0
    
    # Tentukan kondisi server
    if final_result < 0.33:
        condition = "Good"
        condition_class = "success"
    elif 0.33 <= final_result < 0.83:
        condition = "Normal"
        condition_class = "info"
    else:
        condition = "Overheat"
        condition_class = "danger"
    
    return jsonify({
        'temperature': temperature,
        'humidity': humidity,
        'temp_memberships': {k: round(v, 3) for k, v in temp_memberships.items()},
        'hum_memberships': {k: round(v, 3) for k, v in hum_memberships.items()},
        'rule_results': rule_results,
        'final_result': round(final_result, 3),
        'condition': condition,
        'condition_class': condition_class
    })


# GENETIC ALGORITHM - Knapsack Problem
items = {
    'A': {'weight': 7, 'value': 5},
    'B': {'weight': 2, 'value': 4},
    'C': {'weight': 1, 'value': 7},
    'D': {'weight': 9, 'value': 2}
}
capacity = 15
item_list = list(items.keys())
n_items = len(item_list)

def decode(chromosome):
    total_weight = 0
    total_value = 0
    chosen_items = []
    for gene, name in zip(chromosome, item_list):
        if gene == 1:
            total_weight += items[name]['weight']
            total_value += items[name]['value']
            chosen_items.append(name)
    return chosen_items, total_weight, total_value

def fitness(chromosome):
    _, total_weight, total_value = decode(chromosome)
    if total_weight <= capacity:
        return total_value
    else:
        return 0

def roulette_selection(population, fitnesses):
    total_fit = sum(fitnesses)
    if total_fit == 0:
        return random.choice(population)
    pick = random.uniform(0, total_fit)
    current = 0
    for chrom, fit in zip(population, fitnesses):
        current += fit
        if current >= pick:
            return chrom
    return population[-1]

def crossover(p1, p2):
    if len(p1) != len(p2):
        raise ValueError("Parent length mismatch")
    point = random.randint(1, len(p1) - 1)
    child1 = p1[:point] + p2[point:]
    child2 = p2[:point] + p1[point:]
    return child1, child2

def mutate(chromosome, mutation_rate=0.1):
    return [1 - g if random.random() < mutation_rate else g for g in chromosome]

@app.route('/run_genetic_algorithm', methods=['POST'])
def run_genetic_algorithm():
    data = request.get_json()
    pop_size = int(data.get('pop_size', 10))
    generations = int(data.get('generations', 10))
    crossover_rate = float(data.get('crossover_rate', 0.8))
    mutation_rate = float(data.get('mutation_rate', 0.1))
    elitism = True
    
    population = [[random.randint(0, 1) for _ in range(n_items)] for _ in range(pop_size)]
    
    history = []
    
    for gen in range(generations):
        fitnesses = [fitness(ch) for ch in population]
        
        best_index = fitnesses.index(max(fitnesses))
        best_chrom = population[best_index]
        best_fit = fitnesses[best_index]
        best_items, w, v = decode(best_chrom)
        
        history.append({
            'generation': gen + 1,
            'chromosome': best_chrom,
            'items': best_items,
            'weight': w,
            'value': v,
            'fitness': best_fit
        })
        
        new_population = []
        
        if elitism:
            new_population.append(best_chrom)
        
        while len(new_population) < pop_size:
            parent1 = roulette_selection(population, fitnesses)
            parent2 = roulette_selection(population, fitnesses)
            
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        population = new_population[:pop_size]
    
    fitnesses = [fitness(ch) for ch in population]
    best_index = fitnesses.index(max(fitnesses))
    best_chrom = population[best_index]
    best_items, w, v = decode(best_chrom)
    
    return jsonify({
        'history': history,
        'best_chromosome': best_chrom,
        'best_items': best_items,
        'total_weight': w,
        'total_value': v,
        'final_fitness': fitness(best_chrom)
    })


# GENETIC ALGORITHM - TSP (Traveling Salesman Problem)
@app.route('/run_tsp_genetic_algorithm', methods=['POST'])
def run_tsp_genetic_algorithm():
    data = request.get_json()
    
    # Parse distance matrix
    dist_matrix_data = data.get('distance_matrix', [])
    cities_data = data.get('cities', [])
    
    # Convert to numpy array
    dist_matrix = np.array(dist_matrix_data, dtype=float)
    cities = cities_data
    n_cities = len(cities)
    
    # Parameters
    pop_size = int(data.get('pop_size', 100))
    generations = int(data.get('generations', 500))
    tournament_k = int(data.get('tournament_k', 5))
    pc = float(data.get('crossover_rate', 0.9))
    pm = float(data.get('mutation_rate', 0.2))
    elite_size = int(data.get('elite_size', 1))
    
    # Helper functions
    def route_distance(route):
        d = sum(dist_matrix[route[i], route[(i+1) % len(route)]] for i in range(len(route)))
        return d
    
    def create_individual(n):
        ind = list(range(n))
        random.shuffle(ind)
        return ind
    
    def initial_population(size, n):
        return [create_individual(n) for _ in range(size)]
    
    def tournament_selection(pop, k):
        selected = random.sample(pop, k)
        return min(selected, key=lambda ind: route_distance(ind))
    
    def ordered_crossover(p1, p2):
        a, b = sorted(random.sample(range(len(p1)), 2))
        child = [-1] * len(p1)
        child[a:b+1] = p1[a:b+1]
        
        p2_idx = 0
        for i in range(len(p1)):
            if child[i] == -1:
                while p2[p2_idx] in child:
                    p2_idx += 1
                child[i] = p2[p2_idx]
        
        return child
    
    def swap_mutation(ind):
        a, b = random.sample(range(len(ind)), 2)
        ind[a], ind[b] = ind[b], ind[a]
        return ind
    
    # Initialize population
    pop = initial_population(pop_size, n_cities)
    best = min(pop, key=lambda ind: route_distance(ind))
    best_dist = route_distance(best)
    
    history = []
    best_routes_per_gen = []
    
    # Main GA loop
    for g in range(generations):
        new_pop = []
        
        # Sort population
        pop = sorted(pop, key=lambda ind: route_distance(ind))
        
        # Update best
        if route_distance(pop[0]) < best_dist:
            best = pop[0][:]
            best_dist = route_distance(best)
        
        # Elitism
        new_pop.extend([ind[:] for ind in pop[:elite_size]])
        
        # Generate offspring
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, tournament_k)
            p2 = tournament_selection(pop, tournament_k)
            
            child = ordered_crossover(p1, p2) if random.random() < pc else p1[:]
            
            if random.random() < pm:
                child = swap_mutation(child)
            
            new_pop.append(child)
        
        pop = new_pop[:pop_size]
        
        # Record history
        history.append({
            'generation': g + 1,
            'best_distance': round(best_dist, 3)
        })
        
        # Store best route for visualization (every 10 generations or last)
        if g % 10 == 0 or g == generations - 1:
            best_routes_per_gen.append({
                'generation': g + 1,
                'route': [cities[i] for i in best],
                'route_indices': best[:],
                'distance': round(best_dist, 3)
            })
    
    # Final result
    best_route_names = [cities[i] for i in best]
    
    return jsonify({
        'success': True,
        'best_route': best_route_names,
        'best_route_indices': best,
        'best_distance': round(best_dist, 3),
        'history': history,
        'best_routes_per_gen': best_routes_per_gen,
        'cities': cities
    })


if __name__ == '__main__':
    app.run(debug=True)