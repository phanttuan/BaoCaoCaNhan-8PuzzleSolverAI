import pygame
import sys
import heapq
from collections import deque
import time
import random
import math

# Thông số cấu hình
WIDTH, HEIGHT = 1000, 800
TILE_SIZE = 120
BUTTON_WIDTH, BUTTON_HEIGHT = 100, 40
MARGIN = 10

# Màu sắc
WHITE = (255, 255, 255)
DARK_BLUE = (44, 62, 80)
LIGHT_BLUE = (52, 152, 219)
YELLOW = (241, 196, 15)
GRAY = (149, 165, 166)
TEXT_COLOR = (236, 240, 241)
SCROLL_BAR_WIDTH = 20
SCROLL_BAR_COLOR = (180, 220, 240)
SCROLL_THUMB_COLOR = (52, 152, 219)

# Tham số cho Q-learning
Q_LEARNING_EPISODES = 1000
Q_LEARNING_ALPHA = 0.1
Q_LEARNING_GAMMA = 0.9
Q_LEARNING_EPSILON = 0.3
Q_LEARNING_MAX_STEPS_PER_EPISODE = 50
Q_LEARNING_MAX_PATH_LEN = 100

# Khởi tạo pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("23110355_PhanVietTuan")
font = pygame.font.Font(None, 48)
small_font = pygame.font.Font(None, 24)
clock = pygame.time.Clock()

# Khởi tạo puzzle và trạng thái mục tiêu
start_puzzle = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
goal_puzzle = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

# Danh sách các thuật toán
algorithms = [
    # Uninformed Search Algorithms
    "BFS", "DFS", "UCS", "ID", 
    
    # Informed Search Algorithms
    "Greedy", "A*", "IDA*", 
    
    # Local Search Algorithms
    "Simple HC", "Steepest HC", "Stochastic HC", "SA", "Beam", "Genetic",
    
    # Complex Environment Search
    "AND-OR", "Belief State", "PO", 
    
    # Constraint Satisfaction Problems 
    "MC", "BACK", "BACK-FC",

    # Reinforcement Learning
    "Q-Learning"
]

# Khởi tạo biến trạng thái
selected_algorithm = "BFS"
solution = []
step = 0
running = False
paused = False
start_time = 0
elapsed_time = 0

def find_zero(matrix):
    for row in range(3):
        for col in range(3):
            if matrix[row][col] == 0:
                return row, col

def get_moves(matrix):
    zero_row, zero_col = find_zero(matrix)
    moves = []
    for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = zero_row + d_row, zero_col + d_col
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_matrix = [row[:] for row in matrix]
            new_matrix[zero_row][zero_col], new_matrix[new_row][new_col] = new_matrix[new_row][new_col], new_matrix[zero_row][zero_col]
            moves.append(new_matrix)
    return moves

def manhattan_distance(state):
    distance = 0
    goal_positions = {goal_puzzle[r][c]: (r, c) for r in range(3) for c in range(3)}
    for i in range(3):
        for j in range(3):
            value = state[i][j]
            if value != 0:
                goal_x, goal_y = goal_positions[value]
                distance += abs(i - goal_x) + abs(j - goal_y)
    return distance

"""UNINFORMED SEARCH ALGORITHMS"""
def bfs(start):
    queue = deque([(start, [])])
    visited = set()
    while queue:
        current, path = queue.popleft()
        if current == goal_puzzle:
            return path + [current]
        visited.add(tuple(map(tuple, current)))
        for next_state in get_moves(current):
            if tuple(map(tuple, next_state)) not in visited:
                queue.append((next_state, path + [current]))
    return None

def dfs(start, depth_limit=70):
    stack = [(start, [])]
    visited = set()
    while stack:
        current, path = stack.pop()
        if current == goal_puzzle:
            return path + [current]
        if tuple(map(tuple, current)) in visited or len(path) >= depth_limit:
            continue
        visited.add(tuple(map(tuple, current)))
        for next_state in get_moves(current):
            if tuple(map(tuple, next_state)) not in visited:
                stack.append((next_state, path + [current]))
    return None

def ucs(start):
    heap = [(0, start, [])]
    visited = set()
    while heap:
        cost, current, path = heapq.heappop(heap)
        if current == goal_puzzle:
            return path + [current]
        visited.add(tuple(map(tuple, current)))
        for next_state in get_moves(current):
            if tuple(map(tuple, next_state)) not in visited:
                heapq.heappush(heap, (cost + 1, next_state, path + [current]))
    return None
def dls(node, depth, visited):
    if depth == 0:
        return None if node != goal_puzzle else [node]
    if tuple(map(tuple, node)) in visited:
        return None
    visited.add(tuple(map(tuple, node)))
    for move in get_moves(node):
        path = dls(move, depth - 1, visited)
        if path:
            return [node] + path
    return None

def interactive_deepening(start):
    depth = 0
    while depth < 50:
        visited = set()
        result = dls(start, depth, visited)
        if result:
            return result
        depth += 1
    return None

"""HEURISTIC SEARCH ALGORITHMS (informed)"""
def greedy(start):
    heap = [(manhattan_distance(start), start, [])]
    visited = set()
    while heap:
        _, current, path = heapq.heappop(heap)
        if current == goal_puzzle:
            return path + [current]
        visited.add(tuple(map(tuple, current)))
        for next_state in get_moves(current):
            if tuple(map(tuple, next_state)) not in visited:
                heapq.heappush(heap, (manhattan_distance(next_state), next_state, path + [current]))
    return None


def astar(start):
    heap = [(manhattan_distance(start), 0, start, [])]
    visited = set()
    while heap:
        f_cost, g_cost, current, path = heapq.heappop(heap)
        if current == goal_puzzle:
            return path + [current]
        if tuple(map(tuple, current)) in visited:
            continue
        visited.add(tuple(map(tuple, current)))
        for next_state in get_moves(current):
            if tuple(map(tuple, next_state)) not in visited:
                heapq.heappush(heap, (g_cost + 1 + manhattan_distance(next_state), g_cost + 1, next_state, path + [current]))
    return None

def ida_star(start):
    def search(path, g, bound, step):
        node = path[-1]
        f = g + manhattan_distance(node)
        step[0] += 1
        if f > bound:
            return f
        if node == goal_puzzle:
            return path
        min_cost = float('inf')
        for next_state in get_moves(node):
            if next_state not in path:
                path.append(next_state)
                t = search(path, g + 1, bound, step)
                if isinstance(t, list):
                    return t
                if t < min_cost:
                    min_cost = t
                path.pop()
        return min_cost

    bound = manhattan_distance(start)
    path = [start]
    step = [0]
    while True:
        t = search(path, 0, bound, step)
        if isinstance(t, list):
            return t
        if t == float('inf'):
            return None
        bound = t

"""LOCAL SEARCH ALGORITHMS"""
def simple_hill_climbing(start):
    current = start
    path = [current]
    while True:
        if current == goal_puzzle:
            return path
        neighbors = get_moves(current)
        current_distance = manhattan_distance(current)
        for neighbor in neighbors:
            distance = manhattan_distance(neighbor)
            if distance < current_distance:
                current = neighbor
                path.append(current)
                break
        else:
            if current != goal_puzzle:
                return None
            return path

def steepest_ascent_hill_climbing(start):
    current = start
    path = [current]
    while True:
        if current == goal_puzzle:
            return path
        neighbors = get_moves(current)
        best_neighbor = None
        best_distance = manhattan_distance(current)
        for neighbor in neighbors:
            distance = manhattan_distance(neighbor)
            if distance < best_distance:
                best_neighbor = neighbor
                best_distance = distance
        if best_neighbor is None:
            if current != goal_puzzle:
                return None
            return path
        current = best_neighbor
        path.append(current)

def stochastic_hill_climbing(start):
    current = start
    path = [current]
    while True:
        if current == goal_puzzle:
            return path
        neighbors = get_moves(current)
        current_distance = manhattan_distance(current)
        better_neighbors = [n for n in neighbors if manhattan_distance(n) < current_distance]
        if not better_neighbors:
            if current != goal_puzzle:
                return None
            return path
        current = random.choice(better_neighbors)
        path.append(current)

def simulated_annealing(start, initial_temp=1000, cooling_rate=0.95, min_temp=1):
    current = start
    path = [current]
    temperature = initial_temp
    
    while temperature > min_temp:
        if current == goal_puzzle:
            return path
        neighbors = get_moves(current)
        next_state = random.choice(neighbors)
        current_cost = manhattan_distance(current)
        next_cost = manhattan_distance(next_state)
        delta = next_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current = next_state
            path.append(current)
            
        temperature *= cooling_rate
    
    if current != goal_puzzle:
        return None
    return path

def beam_search(start, beam_width=3):
    queue = [(manhattan_distance(start), start, [])]
    while queue:
        next_queue = []
        for _ in range(min(len(queue), beam_width)):
            if not queue:
                break
            _, current, path = heapq.heappop(queue)
            
            if current == goal_puzzle:
                return path + [current]
            
            for next_state in get_moves(current):
                if next_state not in [state for _, state, _ in queue + next_queue]:
                    next_queue.append((manhattan_distance(next_state), next_state, path + [current]))
        
        next_queue.sort(key=lambda x: x[0])
        queue = next_queue[:beam_width]
    
    return None

def genetic_algorithm(start, population_size=100, generations=1000, mutation_rate=0.1):
    def create_individual():
        individual = []
        current = start
        for _ in range(20):
            moves = get_moves(current)
            next_state = random.choice(moves)
            individual.append(next_state)
            current = next_state
        return individual

    def fitness(individual):
        last_state = individual[-1]
        manhattan = manhattan_distance(last_state)
        misplaced = sum(1 for i in range(3) for j in range(3) if last_state[i][j] != goal_puzzle[i][j] and last_state[i][j] != 0)
        return -(manhattan + misplaced)

    def crossover(parent1, parent2):
        cut = random.randint(1, len(parent1) - 1)
        child = parent1[:cut] + parent2[cut:]
        return child

    def mutate(individual):
        if random.random() < mutation_rate:
            index = random.randint(0, len(individual) - 1)
            moves = get_moves(individual[index])
            individual[index] = random.choice(moves)
        return individual

    population = [create_individual() for _ in range(population_size)]

    for generation in range(generations):
        population = sorted(population, key=fitness, reverse=True)
        for individual in population:
            if individual[-1] == goal_puzzle:
                return individual

        next_generation = population[:population_size // 2]

        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(next_generation, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_generation.append(child)

        population = next_generation

    return None

"""COMPLEX ENVIRONMENT SEARCH ALGORITHMS"""
def and_or_tree_search(start, goal, max_depth=50):
    def or_search(state, path, depth, visited):
        if depth > max_depth:
            return None
        if state == goal:
            return path
        state_tuple = tuple(map(tuple, state))
        if state_tuple in visited:
            return None
        visited.add(state_tuple)
        r, c = next((r, c) for r in range(3) for c in range(3) if state[r][c] == 0)
        moves = [(nr, nc) for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)] 
                 if 0 <= nr < 3 and 0 <= nc < 3]
        for move in moves:
            nr, nc = move
            new_state = [row[:] for row in state]
            new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
            result = and_search(new_state, path + [(nr, nc)], depth + 1, visited.copy())
            if result is not None:
                return result
        return None
        
    def and_search(state, path, depth, visited):
        result = or_search(state, path, depth, visited)
        return result
        
    start_time = time.time()
    visited = set()
    path = and_search(start, [], 0, visited)
    elapsed_time = time.time() - start_time
    return path if path is not None else None

def generate_belief_states(goal, count=3):
    """Khởi tạo các trạng thái tin tưởng ngẫu nhiên cho bài toán 8-puzzle"""
    states = []
    for _ in range(count):
        # Khởi tạo một trạng thái ngẫu nhiên
        tiles = list(range(9))
        random.shuffle(tiles)
        state = [[tiles[r*3+c] for c in range(3)] for r in range(3)]
        states.append(state)
    return states

def no_observation_search(initial_states, goal, max_depth=15):
    if not initial_states:
        return None
        
    goal_tuple = tuple(map(tuple, goal))
    belief_state = {tuple(map(tuple, state)): [] for state in initial_states}
    visited = set()
    
    for depth in range(max_depth + 1):
        for state_tuple in belief_state:
            if state_tuple == goal_tuple:
                return belief_state[state_tuple]
        
        if depth == max_depth:
            break
            
        new_belief_state = {}
        
        for state_tuple, path in belief_state.items():
            if state_tuple in visited:
                continue
                
            visited.add(state_tuple)
            state = [list(row) for row in state_tuple]
            
            r, c = next((r, c) for r in range(3) for c in range(3) if state[r][c] == 0)
            
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                if 0 <= nr < 3 and 0 <= nc < 3:
                    new_state = [row[:] for row in state]
                    new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
                    new_state_tuple = tuple(map(tuple, new_state))
                    
                    if new_state_tuple not in visited:
                        new_path = path + [(nr, nc)]
                        new_belief_state[new_state_tuple] = new_path
        
        belief_state = new_belief_state
        
        if not belief_state:
            break

    return None

def partially_observable_search(start, goal, observation_ratio=0.5, max_iterations=1000):
    """
    Giải bài toán 8-puzzle với trạng thái quan sát một phần.
    
    Args:
        start: Trạng thái bắt đầu
        goal: Trạng thái mục tiêu
        observation_ratio: Tỉ lệ quan sát (0.0 - 1.0)
        max_iterations: Số lần lặp tối đa để tìm kiếm
    Returns:
        Danh sách các bước di chuyển để đạt được trạng thái mục tiêu, hoặc None nếu không tìm thấy
    """
    flat_state = [start[r][c] for r in range(3) for c in range(3)]
    
    num_observable = max(1, int(9 * observation_ratio))
    observable_indices = random.sample(range(9), num_observable)
    
    observation_mask = [False] * 9
    for idx in observable_indices:
        observation_mask[idx] = True
    
    masked_state = []
    for r in range(3):
        row = []
        for c in range(3):
            idx = r * 3 + c
            if observation_mask[idx]:
                row.append(start[r][c])
            else:
                row.append("?")
        masked_state.append(row)
    
    possible_states = []
    unknown_positions = [i for i, observed in enumerate(observation_mask) if not observed]
    unknown_values = []
    
    for i in range(9):
        if i not in [flat_state[idx] for idx in observable_indices]:
            unknown_values.append(i)
    
    from itertools import permutations
    for perm in permutations(unknown_values):
        possible_state = flat_state.copy()
        for pos_idx, val_idx in enumerate(range(len(unknown_positions))):
            possible_state[unknown_positions[pos_idx]] = perm[val_idx]
        possible_states.append([[possible_state[r*3+c] for c in range(3)] for r in range(3)])
    
    solution_path = None
    solution_state = None
    visited_states = 0
    
    for possible_start in possible_states:
        queue = deque([(possible_start, [])])
        visited = {tuple(map(tuple, possible_start))}
        visited_states += 1
        steps_explored = 0
        
        while queue and steps_explored < max_iterations:
            state, path = queue.popleft()
            steps_explored += 1
            
            if state == goal:
                solution_path = path
                solution_state = possible_start
                break
                
            r, c = next((r, c) for r in range(3) for c in range(3) if state[r][c] == 0)
            
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                if 0 <= nr < 3 and 0 <= nc < 3:
                    new_state = [row[:] for row in state]
                    new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
                    new_tuple = tuple(map(tuple, new_state))
                    
                    if new_tuple not in visited:
                        queue.append((new_state, path + [(nr, nc)]))
                        visited.add(new_tuple)
                        visited_states += 1
                        
        if solution_path:
            break
    
    return solution_path

"""CSPs SEARCH ALGORITHMS"""
def min_conflicts(start, goal, max_steps=500, use_random_start=True):
    """
    Min-Conflicts algorithm for 8-puzzle (CSPs style).
    Returns a list of states from start to goal, or None if not found.
    """
    def is_solvable(state):
        flat = [cell for row in state for cell in row if cell != 0]
        inv = 0
        for i in range(len(flat)):
            for j in range(i+1, len(flat)):
                if flat[i] > flat[j]:
                    inv += 1
        # Đảm bảo cùng chẵn lẻ với trạng thái đích
        goal_flat = [cell for row in goal for cell in row if cell != 0]
        goal_inv = 0
        for i in range(len(goal_flat)):
            for j in range(i+1, len(goal_flat)):
                if goal_flat[i] > goal_flat[j]:
                    goal_inv += 1
        return inv % 2 == goal_inv % 2

    def manhattan(state):
        goal_pos = {goal[r][c]: (r, c) for r in range(3) for c in range(3)}
        dist = 0
        for r in range(3):
            for c in range(3):
                v = state[r][c]
                if v != 0:
                    gr, gc = goal_pos[v]
                    dist += abs(r - gr) + abs(c - gc)
        return dist

    def get_neighbors(state):
        r0, c0 = next((r, c) for r in range(3) for c in range(3) if state[r][c] == 0)
        neighbors = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r0+dr, c0+dc
            if 0<=nr<3 and 0<=nc<3:
                new_state = [row[:] for row in state]
                new_state[r0][c0], new_state[nr][nc] = new_state[nr][nc], new_state[r0][c0]
                neighbors.append(new_state)
        return neighbors

    def random_state():
        while True:
            flat = list(range(9))
            random.shuffle(flat)
            s = [flat[i*3:(i+1)*3] for i in range(3)]
            if is_solvable(s):
                return s
                
    def random_moves(state, num_moves=3):
        """Thực hiện một số bước đi ngẫu nhiên từ trạng thái hiện tại"""
        current = [row[:] for row in state]
        for _ in range(num_moves):
            neighbors = get_neighbors(current)
            if neighbors:
                current = random.choice(neighbors)
        return current

    # Start from random or given state
    current = random_state() if use_random_start else [row[:] for row in start]
    if not is_solvable(current):
        return None

    path = [current]
    visited = set()
    
    # Ghi nhớ trạng thái tốt nhất đã tìm thấy
    best_state = current
    best_dist = manhattan(current)
    
    # Biến đếm plateau (khi không có tiến bộ)
    plateau_count = 0
    last_dist = best_dist

    for step in range(max_steps):
        if current == goal:
            return path
            
        visited.add(tuple(tuple(row) for row in current))
        neighbors = get_neighbors(current)
        
        # Min-conflict: pick neighbor with least conflicts (manhattan)
        min_conf = float('inf')
        best_neighbors = []
        
        for n in neighbors:
            n_dist = manhattan(n)
            if n_dist < min_conf:
                min_conf = n_dist
                best_neighbors = [n]
            elif n_dist == min_conf:
                best_neighbors.append(n)
        
        # Chọn ngẫu nhiên từ các lựa chọn tốt nhất
        next_state = random.choice(best_neighbors)
        
        # Tránh chu trình
        next_tuple = tuple(tuple(row) for row in next_state)
        if next_tuple in visited:
            # Thử khởi động lại từ trạng thái tốt nhất với một số bước đi ngẫu nhiên
            current = random_moves(best_state)
            path.append(current)
            visited = {tuple(tuple(row) for row in current)}
            plateau_count = 0
            continue
            
        # Cập nhật trạng thái tốt nhất nếu cần
        current_dist = manhattan(next_state)
        if current_dist < best_dist:
            best_dist = current_dist
            best_state = [row[:] for row in next_state]
            plateau_count = 0
        elif current_dist == last_dist:
            plateau_count += 1
            
        # Khởi động lại nếu bị kẹt quá lâu
        if plateau_count > 10:
            if random.random() < 0.7:  # 70% cơ hội khởi động lại
                current = random_moves(best_state)
                path.append(current)
                visited = {tuple(tuple(row) for row in current)}
                plateau_count = 0
                continue
                
        current = next_state
        path.append(current)
        last_dist = current_dist
        
    return None

def backtracking_search(start, max_depth=15):
    empty_state = [[None for _ in range(3)] for _ in range(3)]
    used_values = set()
    all_states = []
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    
    cells_order = [(r, c) for r in range(3) for c in range(3)]
    random.shuffle(cells_order)
    
    goal_positions = {}
    for r in range(3):
        for c in range(3):
            goal_positions[goal[r][c]] = (r, c)
    
    def is_valid_assignment(state, r, c, value):
        if value in used_values:
            return False
        return True
    
    def count_inversions(state):
        flat = []
        for i in range(3):
            for j in range(3):
                if state[i][j] is not None and state[i][j] != 0:
                    flat.append(state[i][j])
                    
        inversions = 0
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                if flat[i] > flat[j]:
                    inversions += 1
        return inversions
    
    def is_valid_state(state):
        filled_count = sum(1 for r in range(3) for c in range(3) 
                          if state[r][c] is not None)
        if filled_count == 8:
            empty_pos = next((r, c) for r in range(3) for c in range(3) 
                           if state[r][c] is None)
            
            complete_state = [row[:] for row in state]
            complete_state[empty_pos[0]][empty_pos[1]] = 0
            
            inversions = count_inversions(complete_state)
            if inversions % 2 != 0:
                return False
            
        return True
    
    def recursive_backtracking(state, cell_idx):
        nonlocal all_states
        
        if cell_idx >= len(cells_order):
            current_display = [[0 if cell is None else cell for cell in row] for row in state]
            return current_display == goal
            
        r, c = cells_order[cell_idx]
        target_value = goal[r][c]
        values_to_try = [target_value]

        if target_value in used_values:
            values_to_try = [v for v in range(9) if v not in used_values]
            random.shuffle(values_to_try)
        
        for value in values_to_try:
            if is_valid_assignment(state, r, c, value):
                state[r][c] = value
                used_values.add(value)
                
                current_display = [[0 if cell is None else cell for cell in row] for row in state]
                all_states.append(current_display[:])
                
                if is_valid_state(state) and recursive_backtracking(state, cell_idx + 1):
                    return True
                state[r][c] = None
                used_values.remove(value)
        
        return False
    
    recursive_backtracking(empty_state, 0)
    return all_states

def backtracking_with_forward_checking(start, max_depth=15):
    """
    Giải bài toán 8-puzzle bằng backtracking with forward checking.
    
    Args:
        start: Trạng thái bắt đầu
        max_depth: Độ sâu tối đa cho tìm kiếm
        
    Returns:
        Một danh sách các trạng thái tạo thành lời giải
    """
    empty_state = [[None for _ in range(3)] for _ in range(3)]
    all_states = []
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    
    # Xáo trộn thứ tự duyệt các ô để tạo sự đa dạng
    cells_order = [(r, c) for r in range(3) for c in range(3)]
    random.shuffle(cells_order)
    
    # Domains là miền giá trị khả thi cho mỗi ô
    domains = {}
    for r, c in cells_order:
        domains[(r, c)] = list(range(9))  # Tất cả giá trị từ 0-8 đều có thể
    
    def count_inversions(state):
        flat = []
        for i in range(3):
            for j in range(3):
                if state[i][j] is not None and state[i][j] != 0:
                    flat.append(state[i][j])
                    
        inversions = 0
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                if flat[i] > flat[j]:
                    inversions += 1
        return inversions
    
    def is_valid_state(state):
        filled_count = sum(1 for r in range(3) for c in range(3) if state[r][c] is not None)
        if filled_count == 8:
            empty_pos = next((r, c) for r in range(3) for c in range(3) if state[r][c] is None)
            
            complete_state = [row[:] for row in state]
            complete_state[empty_pos[0]][empty_pos[1]] = 0
            
            inversions = count_inversions(complete_state)
            if inversions % 2 != 0:
                return False
            
        return True
    
    def update_domains(state, domains, r, c, value):
        """
        Cập nhật các miền giá trị sau khi gán giá trị value cho ô (r, c)
        Trả về các domains đã cập nhật và True nếu tất cả miền vẫn khả thi
        """
        new_domains = {pos: list(values) for pos, values in domains.items()}
        
        # Xóa giá trị đã được sử dụng khỏi domains của các ô khác
        for pos in new_domains:
            if pos != (r, c) and value in new_domains[pos]:
                new_domains[pos].remove(value)
                
            # Nếu một ô không có giá trị khả thi, forward checking thất bại
            if pos != (r, c) and state[pos[0]][pos[1]] is None and not new_domains[pos]:
                return new_domains, False
                
        return new_domains, True
    
    def recursive_backtracking_fc(state, domains, cell_idx):
        nonlocal all_states
        
        # Nếu đã gán giá trị cho tất cả các ô
        if cell_idx >= len(cells_order):
            current_display = [[0 if cell is None else cell for cell in row] for row in state]
            return current_display == goal
            
        r, c = cells_order[cell_idx]
        target_value = goal[r][c]
        
        # Ưu tiên giá trị mục tiêu nếu có thể
        values_to_try = []
        if target_value in domains[(r, c)]:
            values_to_try = [target_value]
        else:
            values_to_try = domains[(r, c)].copy()
            random.shuffle(values_to_try)
        
        for value in values_to_try:
            # Gán giá trị cho ô hiện tại
            state[r][c] = value
            
            # Lưu trạng thái hiện tại để hiển thị
            current_display = [[0 if cell is None else cell for cell in row] for row in state]
            all_states.append(current_display[:])
            
            # Kiểm tra tính hợp lệ của trạng thái
            if is_valid_state(state):
                # Forward checking: cập nhật domains
                new_domains, domains_consistent = update_domains(state, domains, r, c, value)
                
                if domains_consistent:
                    # Tiếp tục backtracking với domains đã cập nhật
                    if recursive_backtracking_fc(state, new_domains, cell_idx + 1):
                        return True
            
            # Backtrack nếu không tìm thấy lời giải
            state[r][c] = None
        
        return False
    
    # Bắt đầu backtracking với forward checking
    recursive_backtracking_fc(empty_state, domains, 0)
    return all_states

"""REINFORCEMENT LEARNING ALGORITHMS"""
def state_to_tuple(state):
    """Biên đổi trạng thái từ danh sách 2D sang tuple để sử dụng trong Q-table"""
    return tuple(tuple(row) for row in state)
def path_to_states(start_state, move_path):
    """Biến đổi danh sách các bước di chuyển thành danh sách trạng thái"""
    if not move_path:
        return None
        
    states = [start_state]
    current = [row[:] for row in start_state]
    
    for move in move_path:
        r, c = next((r, c) for r in range(3) for c in range(3) if current[r][c] == 0)
        nr, nc = move
        new_state = [row[:] for row in current]
        new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
        states.append(new_state)
        current = new_state
        
    return states
def get_valid_actions_q(state_tuple):
    """Kiếm tra các hành động hợp lệ từ trạng thái hiện tại"""
    state = [list(row) for row in state_tuple]
    # Tìm vị trí của ô trống (0)
    blank_row, blank_col = None, None
    for r in range(3):
        for c in range(3):
            if state[r][c] == 0:
                blank_row, blank_col = r, c
                break
        if blank_row is not None:
            break
    
    valid_actions = []
    # Check lên
    if blank_row > 0:
        valid_actions.append(0)
    # Check xuống
    if blank_row < 2:
        valid_actions.append(1)
    # Check trái
    if blank_col > 0:
        valid_actions.append(2)
    # Check phải
    if blank_col < 2:
        valid_actions.append(3)
        
    return valid_actions

def apply_action_q(state_tuple, action):
    """Thực hiện hành động trên trạng thái hiện tại và trả về trạng thái mới"""
    state = [list(row) for row in state_tuple]
    # Tìm vị trí của ô trống (0)
    blank_row, blank_col = None, None
    for r in range(3):
        for c in range(3):
            if state[r][c] == 0:
                blank_row, blank_col = r, c
                break
        if blank_row is not None:
            break
    
    # Tính toán vị trí mới của ô trống sau khi thực hiện hành động
    new_blank_row, new_blank_col = blank_row, blank_col
    
    if action == 0:  # lên
        new_blank_row = blank_row - 1
    elif action == 1:  # xuống
        new_blank_row = blank_row + 1
    elif action == 2:  # trái
        new_blank_col = blank_col - 1
    elif action == 3:  # phải
        new_blank_col = blank_col + 1
    
    # Thực hiện hoán đổi ô trống với ô bên cạnh
    state[blank_row][blank_col], state[new_blank_row][new_blank_col] = state[new_blank_row][new_blank_col], state[blank_row][blank_col]
    
    return state_to_tuple(state), (new_blank_row, new_blank_col)

def q_learning_solve(start, goal, episodes=Q_LEARNING_EPISODES, alpha=Q_LEARNING_ALPHA, gamma=Q_LEARNING_GAMMA, 
                     epsilon_start=Q_LEARNING_EPSILON, max_steps_episode=Q_LEARNING_MAX_STEPS_PER_EPISODE, max_path_len=Q_LEARNING_MAX_PATH_LEN):
    """
    Sử dụng Q-learning để giải bài toán 8-puzzle.
    
    Args:
        start: Trạng thái bắt đầu
        goal: Trạng thái mục tiêu
        episodes: Số lượng tập huấn luyện
        alpha: Tốc độ học
        gamma: Hệ số giảm giá
        epsilon_start: Tỉ lệ khám phá ban đầu
        max_steps_episode: Số bước tối đa trong mỗi tập huấn luyện
        max_path_len: Độ dài tối đa của đường đi tìm được
        
    Returns:
        Một danh sách các bước di chuyển để đạt được trạng thái mục tiêu, hoặc None nếu không tìm thấy
    """
    start_time_total = time.time()
    q_table = {} 
    
    start_tuple = state_to_tuple(start)
    goal_tuple = state_to_tuple(goal)

    # Train Q-learning agent
    for episode in range(episodes):
        current_state_tuple = start_tuple 
        epsilon = epsilon_start * math.exp(-episode / (episodes/5))
        for _step in range(max_steps_episode):
            valid_actions = get_valid_actions_q(current_state_tuple)
            if not valid_actions:
                break

            current_q_values = q_table.get(current_state_tuple, [0.0] * 4)  # Khởi tạo Q-values nếu chưa có
            action_to_take = -1

            if random.random() < epsilon:
                action_to_take = random.choice(valid_actions)  # Exploration: chọn hành động ngẫu nhiên
            else:
                # Exploitation: chọn hành động tốt nhất từ Q-table
                best_q_val = -float('inf')
                shuffled_valid_actions = random.sample(valid_actions, len(valid_actions))
                for act_idx in shuffled_valid_actions: 
                    if current_q_values[act_idx] > best_q_val:
                        best_q_val = current_q_values[act_idx]
                        action_to_take = act_idx
                if action_to_take == -1: 
                    action_to_take = random.choice(valid_actions)
            next_state_tuple, _ = apply_action_q(current_state_tuple, action_to_take)
            
            reward = -1
            if next_state_tuple == goal_tuple:
                reward = 100
            
            old_q_value = current_q_values[action_to_take]
            
            # tính toán giá trị Q cho trạng thái tiếp theo
            next_valid_actions = get_valid_actions_q(next_state_tuple)
            next_q_state_values = q_table.get(next_state_tuple, [0.0] * 4)
            max_next_q = 0.0 
            if next_valid_actions:
                max_next_q = max([next_q_state_values[next_act_idx] for next_act_idx in next_valid_actions])

            new_q_value = old_q_value + alpha * (reward + gamma * max_next_q - old_q_value)
            
            updated_q_values = list(current_q_values)
            updated_q_values[action_to_take] = new_q_value
            q_table[current_state_tuple] = updated_q_values
            
            current_state_tuple = next_state_tuple
            if current_state_tuple == goal_tuple:
                break
    
    path = []
    current_state_tuple = start_tuple
    for _ in range(max_path_len):
        if current_state_tuple == goal_tuple:
            break
        
        valid_actions = get_valid_actions_q(current_state_tuple)
        if not valid_actions:
            path = []
            break

        current_q_values = q_table.get(current_state_tuple, [0.0] * 4)
        
        best_action = max(valid_actions, key=lambda x: current_q_values[x])
        
        next_state_tuple, new_blank_coords = apply_action_q(current_state_tuple, best_action)
        path.append(new_blank_coords)
        current_state_tuple = next_state_tuple
    
    else:
        if current_state_tuple != goal_tuple:
            path = []

    elapsed_time = time.time() - start_time_total
    return path, elapsed_time

def q_learning(start):
    """Giải bài toán 8-puzzle bằng Q-learning."""
    path, _ = q_learning_solve(start, goal_puzzle)
    if not path:
        return None
    return path_to_states(start, path)

"""GUI và điều khiển"""
def draw_puzzle(matrix, selected_tile=None):
    screen.fill((192, 240, 255))

    title = font.render("8-Puzzle Solver", True, (12, 180, 178))
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, MARGIN))

    board_size = 3 * TILE_SIZE
    board_x = (WIDTH - board_size) // 2
    board_y = HEIGHT // 2 - board_size // 2 + 30
    pygame.draw.rect(screen, (36, 217, 227), (board_x, board_y, board_size, board_size))

    for row in range(3):
        for col in range(3):
            value = matrix[row][col]
            x = board_x + col * TILE_SIZE
            y = board_y + row * TILE_SIZE

            pygame.draw.rect(screen, (12, 180, 178), (x + 5, y + 5, TILE_SIZE - 10, TILE_SIZE - 10))
            pygame.draw.rect(screen, (0, 0, 0), (x + 5, y + 5, TILE_SIZE - 10, TILE_SIZE - 10), 3)

            if value != 0:
                text = font.render(str(value), True, WHITE)
                text_rect = text.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2))
                screen.blit(text, text_rect)

            if selected_tile == (row, col):
                pygame.draw.rect(screen, YELLOW, (x + 5, y + 5, TILE_SIZE - 10, TILE_SIZE - 10), 3)

    button_y_top = MARGIN + 80
    button_y_bottom = button_y_top + BUTTON_HEIGHT + MARGIN
    algo_per_row = 9
    total_buttons_width = algo_per_row * (BUTTON_WIDTH + MARGIN) - MARGIN
    start_x = (WIDTH - total_buttons_width) // 2
    mouse_x, mouse_y = pygame.mouse.get_pos()

    for i, algo in enumerate(algorithms):
        x = start_x + (i % algo_per_row) * (BUTTON_WIDTH + MARGIN)
        y = button_y_top + (i // algo_per_row) * (BUTTON_HEIGHT + MARGIN)
        color = (36, 217, 227) if algo == selected_algorithm else (12, 180, 178)
        if x <= mouse_x <= x + BUTTON_WIDTH and y <= mouse_y <= y + BUTTON_HEIGHT:
            color = (70, 180, 240)

        pygame.draw.rect(screen, color, (x, y, BUTTON_WIDTH, BUTTON_HEIGHT))
        pygame.draw.rect(screen, (0, 0, 0), (x, y, BUTTON_WIDTH, BUTTON_HEIGHT), 3)

        text = small_font.render(algo if "HC" not in algo else algo.replace(" HC", ""), True, WHITE)
        text_rect = text.get_rect(center=(x + BUTTON_WIDTH // 2, y + BUTTON_HEIGHT // 2))
        screen.blit(text, text_rect)

    control_y = HEIGHT - 120
    buttons = [
        ("Start", (100, 200, 100)),
        ("Pause", (240, 180, 50)),
        ("Reset", (200, 80, 80)),
        ("Steps", (100, 100, 200))
    ]
    total_control_width = len(buttons) * (BUTTON_WIDTH + MARGIN) - MARGIN
    control_x = (WIDTH - total_control_width) // 2

    for i, (text, color) in enumerate(buttons):
        x = control_x + i * (BUTTON_WIDTH + MARGIN)
        if x <= mouse_x <= x + BUTTON_WIDTH and control_y <= mouse_y <= control_y + BUTTON_HEIGHT:
            pygame.draw.rect(screen, (50, 50, 50), (x + 2, control_y + 2, BUTTON_WIDTH, BUTTON_HEIGHT))
        pygame.draw.rect(screen, color, (x, control_y, BUTTON_WIDTH, BUTTON_HEIGHT))
        txt = small_font.render(text, True, WHITE)
        txt_rect = txt.get_rect(center=(x + BUTTON_WIDTH // 2, control_y + BUTTON_HEIGHT // 2))
        screen.blit(txt, txt_rect)

    timer_bg = pygame.Surface((200, 40), pygame.SRCALPHA)
    timer_bg.fill((200, 200, 255, 180))
    screen.blit(timer_bg, (MARGIN, HEIGHT - 40))
    timer_text = small_font.render(f"Time: {elapsed_time:.4f}s", True, DARK_BLUE)
    screen.blit(timer_text, (MARGIN + 10, HEIGHT - 30))


def show_steps_window(solution):
    popup_width, popup_height = 500, 800
    popup_surface = pygame.Surface((popup_width, popup_height))
    popup_rect = popup_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    
    tile_size = 60
    step_height = tile_size * 3 + 40
    scroll_y = 0
    max_scroll = max(0, len(solution) * step_height - (popup_height - 100))
    
    dragging = False
    last_mouse_y = 0
    running = True

    def get_changed_tiles(state1, state2):
        changed = []
        for row in range(3):
            for col in range(3):
                if state1[row][col] != state2[row][col]:
                    changed.append((row, col))
        return changed

    while running:
        popup_surface.fill((192, 240, 255))

        pygame.draw.rect(popup_surface, DARK_BLUE, (0, 0, popup_width, popup_height), 5)
        
        title = font.render("Solution Steps", True, DARK_BLUE)
        popup_surface.blit(title, (popup_width // 2 - title.get_width() // 2, 20))

        close_button_rect = pygame.Rect(popup_width - 90, 10, 70, 30)
        pygame.draw.rect(popup_surface, (200, 50, 50), close_button_rect)
        pygame.draw.rect(popup_surface, DARK_BLUE, close_button_rect, 2)
        close_text = small_font.render("Close", True, WHITE)
        popup_surface.blit(close_text, (close_button_rect.centerx - close_text.get_width() // 2,
                                        close_button_rect.centery - close_text.get_height() // 2))

        for i, state in enumerate(solution):
            y = 80 + i * step_height - scroll_y
            if y < 50 or y > popup_height - step_height:
                continue

            step_text = small_font.render(f"Step {i + 1}", True, DARK_BLUE)
            popup_surface.blit(step_text, (20, y))

            board_x = popup_width // 2 - (tile_size * 3) // 2
            changed_tiles = get_changed_tiles(solution[i - 1], state) if i > 0 else []
            for row in range(3):
                for col in range(3):
                    value = state[row][col]
                    x = board_x + col * tile_size
                    y_pos = y + row * tile_size + 20
                    color = (255, 100, 100) if (row, col) in changed_tiles else (12, 180, 178)
                    pygame.draw.rect(popup_surface, color, (x + 5, y_pos + 5, tile_size - 10, tile_size - 10))
                    pygame.draw.rect(popup_surface, (0, 0, 0), (x + 5, y_pos + 5, tile_size - 10, tile_size - 10), 3)
                    if value != 0:
                        text = small_font.render(str(value), True, WHITE)
                        text_rect = text.get_rect(center=(x + tile_size // 2, y_pos + tile_size // 2))
                        popup_surface.blit(text, text_rect)

        if max_scroll > 0:
            scroll_bar_height = (popup_height - 100) * (popup_height - 100) / (len(solution) * step_height)
            scroll_thumb_y = 80 + (scroll_y / max_scroll) * (popup_height - 100 - scroll_bar_height)
            pygame.draw.rect(popup_surface, SCROLL_BAR_COLOR, 
                             (popup_width - SCROLL_BAR_WIDTH - 10, 80, SCROLL_BAR_WIDTH, popup_height - 100))
            pygame.draw.rect(popup_surface, SCROLL_THUMB_COLOR, 
                             (popup_width - SCROLL_BAR_WIDTH - 10, scroll_thumb_y, SCROLL_BAR_WIDTH, scroll_bar_height))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                x -= popup_rect.left
                y -= popup_rect.top
                if close_button_rect.collidepoint(x, y):
                    running = False
                elif popup_width - SCROLL_BAR_WIDTH - 10 <= x <= popup_width - 10 and 80 <= y <= popup_height - 20:
                    dragging = True
                    last_mouse_y = y
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                x, y = event.pos
                y -= popup_rect.top
                delta_y = y - last_mouse_y
                scroll_y += delta_y * (max_scroll / (popup_height - 100))
                scroll_y = max(0, min(scroll_y, max_scroll))
                last_mouse_y = y
            elif event.type == pygame.MOUSEWHEEL:
                scroll_y -= event.y * 50
                scroll_y = max(0, min(scroll_y, max_scroll))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        screen.blit(popup_surface, popup_rect)
        pygame.display.flip()
        clock.tick(60)


def show_belief_states_popup(goal_puzzle):
    """Hiển thị cửa sổ chọn trạng thái tin tưởng"""
    belief_states = generate_belief_states(goal_puzzle, count=3)
    
    popup_width, popup_height = 850, 750
    popup_surface = pygame.Surface((popup_width, popup_height))
    popup_rect = popup_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    
    tile_size = 70
    state_height = tile_size * 3 + 80
    scroll_y = 0
    max_scroll = max(0, len(belief_states) * state_height - (popup_height - 200))
    
    dragging = False
    last_mouse_y = 0
    selected_state_idx = None
    custom_state = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    editing_custom = False
    selected_cell = None
    
    running = True

    solve_button_rect = pygame.Rect(popup_width - 200, popup_height - 60, 180, 40)
    close_button_rect = pygame.Rect(popup_width - 90, 10, 70, 30)
    random_button_rect = pygame.Rect(20, popup_height - 60, 180, 40)
    custom_button_rect = pygame.Rect(220, popup_height - 60, 180, 40)
    
    while running:
        popup_surface.fill((192, 240, 255))

        pygame.draw.rect(popup_surface, DARK_BLUE, (0, 0, popup_width, popup_height), 5)
        
        title = font.render("Select Belief State", True, DARK_BLUE)
        popup_surface.blit(title, (popup_width // 2 - title.get_width() // 2, 20))
        
        info_text = small_font.render("Select one state or use the options below", True, DARK_BLUE)
        popup_surface.blit(info_text, (20, 60))

        # Vẽ nút close
        pygame.draw.rect(popup_surface, (200, 50, 50), close_button_rect)
        pygame.draw.rect(popup_surface, DARK_BLUE, close_button_rect, 2)
        close_text = small_font.render("Close", True, WHITE)
        popup_surface.blit(close_text, (close_button_rect.centerx - close_text.get_width() // 2,
                                       close_button_rect.centery - close_text.get_height() // 2))

        # Vẽ nút solve
        button_color = (100, 200, 100) if selected_state_idx is not None or editing_custom else (150, 150, 150)
        pygame.draw.rect(popup_surface, button_color, solve_button_rect)
        pygame.draw.rect(popup_surface, DARK_BLUE, solve_button_rect, 2)
        solve_text = font.render("Solve", True, WHITE)
        popup_surface.blit(solve_text, (solve_button_rect.centerx - solve_text.get_width() // 2,
                                       solve_button_rect.centery - solve_text.get_height() // 2))
                                       
        # Vẽ nút random
        pygame.draw.rect(popup_surface, (100, 100, 200), random_button_rect)
        pygame.draw.rect(popup_surface, DARK_BLUE, random_button_rect, 2)
        random_text = small_font.render("Generate Random", True, WHITE)
        popup_surface.blit(random_text, (random_button_rect.centerx - random_text.get_width() // 2,
                                       random_button_rect.centery - random_text.get_height() // 2))
                                       
        # Vẽ nút custom
        custom_color = (36, 217, 227) if editing_custom else (12, 180, 178)
        pygame.draw.rect(popup_surface, custom_color, custom_button_rect)
        pygame.draw.rect(popup_surface, DARK_BLUE, custom_button_rect, 2)
        custom_text = small_font.render("Create Custom", True, WHITE)
        popup_surface.blit(custom_text, (custom_button_rect.centerx - custom_text.get_width() // 2,
                                      custom_button_rect.centery - custom_text.get_height() // 2))

        if editing_custom:
            custom_y = 120
            custom_text = font.render("Create your own belief state:", True, DARK_BLUE)
            popup_surface.blit(custom_text, (20, custom_y))
            
            custom_y += 50
            board_x = popup_width // 2 - (tile_size * 3) // 2
            
            for row in range(3):
                for col in range(3):
                    value = custom_state[row][col]
                    x = board_x + col * tile_size
                    y_pos = custom_y + row * tile_size
                    
                    if selected_cell == (row, col):
                        color = (255, 220, 100)
                    else:
                        color = (12, 180, 178)
                        
                    pygame.draw.rect(popup_surface, color, (x + 5, y_pos + 5, tile_size - 10, tile_size - 10))
                    pygame.draw.rect(popup_surface, (0, 0, 0), (x + 5, y_pos + 5, tile_size - 10, tile_size - 10), 3)
                    
                    if value != 0:
                        text = font.render(str(value), True, WHITE)
                        text_rect = text.get_rect(center=(x + tile_size // 2, y_pos + tile_size // 2))
                        popup_surface.blit(text, text_rect)
            
            instr_y = custom_y + 3 * tile_size + 30
            instr_text = small_font.render("Choose 1 tile and enter a number.", True, DARK_BLUE)
            popup_surface.blit(instr_text, (20, instr_y))
            
        else:
            for i, state in enumerate(belief_states):
                y = 120 + i * state_height - scroll_y
                if y < 100 or y > popup_height - 100:
                    continue

                radio_rect = pygame.Rect(20, y + tile_size, 20, 20)
                pygame.draw.rect(popup_surface, WHITE, radio_rect)
                pygame.draw.rect(popup_surface, DARK_BLUE, radio_rect, 2)
                
                if i == selected_state_idx:
                    pygame.draw.circle(popup_surface, DARK_BLUE, 
                                    (radio_rect.centerx, radio_rect.centery), 7)

                state_text = small_font.render(f"State {i+1}", True, DARK_BLUE)
                popup_surface.blit(state_text, (50, y + tile_size))

                board_x = popup_width // 2 - (tile_size * 3) // 2
                for row in range(3):
                    for col in range(3):
                        value = state[row][col]
                        x = board_x + col * tile_size
                        y_pos = y + row * tile_size
                        color = (12, 180, 178)
                        pygame.draw.rect(popup_surface, color, (x + 5, y_pos + 5, tile_size - 10, tile_size - 10))
                        pygame.draw.rect(popup_surface, (0, 0, 0), (x + 5, y_pos + 5, tile_size - 10, tile_size - 10), 3)
                        if value != 0:
                            text = font.render(str(value), True, WHITE)
                            text_rect = text.get_rect(center=(x + tile_size // 2, y_pos + tile_size // 2))
                            popup_surface.blit(text, text_rect)

                pygame.draw.line(popup_surface, (150, 150, 150), 
                                (20, y + state_height - 30), 
                                (popup_width - 20, y + state_height - 30), 1)

            if max_scroll > 0:
                scroll_bar_height = (popup_height - 220) * (popup_height - 220) / (len(belief_states) * state_height)
                scroll_thumb_y = 100 + (scroll_y / max_scroll) * (popup_height - 220 - scroll_bar_height)
                pygame.draw.rect(popup_surface, SCROLL_BAR_COLOR, 
                                (popup_width - SCROLL_BAR_WIDTH - 10, 100, SCROLL_BAR_WIDTH, popup_height - 220))
                pygame.draw.rect(popup_surface, SCROLL_THUMB_COLOR, 
                                (popup_width - SCROLL_BAR_WIDTH - 10, scroll_thumb_y, SCROLL_BAR_WIDTH, scroll_bar_height))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                x -= popup_rect.left
                y -= popup_rect.top
                
                if close_button_rect.collidepoint(x, y):
                    running = False
                    return None
                
                elif solve_button_rect.collidepoint(x, y):
                    if editing_custom:
                        flat_custom = [custom_state[r][c] for r in range(3) for c in range(3)]
                        if sorted(flat_custom) == list(range(9)):
                            return [custom_state]
                    elif selected_state_idx is not None:
                        return [belief_states[selected_state_idx]]
                
                elif random_button_rect.collidepoint(x, y):
                    new_state = generate_belief_states(goal_puzzle, count=1)[0]
                    return [new_state]
                    
                elif custom_button_rect.collidepoint(x, y):
                    editing_custom = not editing_custom
                    selected_state_idx = None if editing_custom else selected_state_idx
                    selected_cell = None
                
                elif editing_custom:
                    board_x = popup_width // 2 - (tile_size * 3) // 2
                    custom_y = 170
                    
                    for row in range(3):
                        for col in range(3):
                            tile_x = board_x + col * tile_size
                            tile_y = custom_y + row * tile_size
                            
                            if (tile_x <= x <= tile_x + tile_size and 
                                tile_y <= y <= tile_y + tile_size):
                                selected_cell = (row, col)
                                break
                
                elif popup_width - SCROLL_BAR_WIDTH - 10 <= x <= popup_width - 10 and 100 <= y <= popup_height - 120:
                    dragging = True
                    last_mouse_y = y
                
                else:
                    for i, state in enumerate(belief_states):
                        radio_y = 120 + i * state_height - scroll_y + tile_size
                        radio_rect = pygame.Rect(20, radio_y, 20, 20)
                        if radio_rect.collidepoint(x, y):
                            selected_state_idx = i
                            editing_custom = False
                            break
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
                
            elif event.type == pygame.MOUSEMOTION and dragging:
                x, y = event.pos
                y -= popup_rect.top
                delta_y = y - last_mouse_y
                scroll_y += delta_y * (max_scroll / (popup_height - 220))
                scroll_y = max(0, min(scroll_y, max_scroll))
                last_mouse_y = y
                
            elif event.type == pygame.MOUSEWHEEL and not editing_custom:
                scroll_y -= event.y * 50
                scroll_y = max(0, min(scroll_y, max_scroll))
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    return None
                    
                if editing_custom and selected_cell is not None:
                    r, c = selected_cell
                    if pygame.K_0 <= event.key <= pygame.K_8:
                        value = event.key - pygame.K_0
                        
                        flat_custom = [custom_state[i][j] for i in range(3) for j in range(3) 
                                     if (i,j) != (r,c)]
                                     
                        if value not in flat_custom:
                            custom_state[r][c] = value
        
        screen.blit(popup_surface, popup_rect)
        pygame.display.flip()
        clock.tick(60)
    
    return None

def show_po_settings_popup():
    """Hiển thị cửa sổ chọn tỷ lệ quan sát"""
    popup_width, popup_height = 500, 400 
    popup_surface = pygame.Surface((popup_width, popup_height))
    popup_rect = popup_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    
    observation_ratio = 0.5
    dragging_slider = False
    
    slider_width = 300
    slider_x = (popup_width - slider_width) // 2
    slider_y = popup_height // 4
    
    apply_button_rect = pygame.Rect(popup_width // 2 - 100, popup_height - 60, 200, 40)
    close_button_rect = pygame.Rect(popup_width - 90, 10, 70, 30)
    
    running = True
    
    while running:
        popup_surface.fill((192, 240, 255))
        pygame.draw.rect(popup_surface, DARK_BLUE, (0, 0, popup_width, popup_height), 5)
        
        title = small_font.render("Partially Observable", True, DARK_BLUE)
        popup_surface.blit(title, (popup_width // 2 - title.get_width() // 2, 20))
        
        info_text = small_font.render("Set the possibility",True, DARK_BLUE)
        popup_surface.blit(info_text, (popup_width // 2 - info_text.get_width() // 2, 70))
        
        # vẽ nút close
        pygame.draw.rect(popup_surface, (200, 50, 50), close_button_rect)
        pygame.draw.rect(popup_surface, DARK_BLUE, close_button_rect, 2)
        close_text = small_font.render("Close", True, WHITE)
        popup_surface.blit(close_text, (close_button_rect.centerx - close_text.get_width() // 2,
                                       close_button_rect.centery - close_text.get_height() // 2))
        
        pygame.draw.rect(popup_surface, (150, 150, 150), 
                        (slider_x, slider_y, slider_width, 10))
        
        thumb_x = slider_x + int(observation_ratio * slider_width)
        pygame.draw.circle(popup_surface, (36, 217, 227), (thumb_x, slider_y + 5), 15)
        pygame.draw.circle(popup_surface, DARK_BLUE, (thumb_x, slider_y + 5), 15, 2)
        
        value_text = font.render(f"{int(observation_ratio * 100)}%", True, DARK_BLUE)
        popup_surface.blit(value_text, (popup_width // 2 - value_text.get_width() // 2, slider_y + 40))
        
        example_y = slider_y + 100
        example_tile_size = 30
        
        for i, example_ratio in enumerate([0.25, 0.5, 0.75]):
            is_selected = abs(observation_ratio - example_ratio) < 0.125
            
            x_pos = popup_width // 4 * (i + 1) - example_tile_size * 1.5
            
            if is_selected:
                pygame.draw.rect(popup_surface, (255, 255, 150), 
                                (x_pos - 5, example_y - 5, 
                                 example_tile_size * 3 + 10, example_tile_size * 3 + 10))
            
            for row in range(3):
                for col in range(3):
                    cell_x = x_pos + col * example_tile_size
                    cell_y = example_y + row * example_tile_size
                    
                    random.seed(i*100 + row*10 + col)
                    is_visible = random.random() < example_ratio
                    
                    color = (12, 180, 178) if is_visible else (100, 100, 100)
                    pygame.draw.rect(popup_surface, color, 
                                    (cell_x, cell_y, example_tile_size - 2, example_tile_size - 2))
            
            label = small_font.render(f"{int(example_ratio * 100)}%", True, DARK_BLUE)
            popup_surface.blit(label, (x_pos + example_tile_size * 1.5 - label.get_width() // 2, 
                                      example_y + example_tile_size * 3 + 15))
        
        pygame.draw.rect(popup_surface, (100, 200, 100), apply_button_rect)
        pygame.draw.rect(popup_surface, DARK_BLUE, apply_button_rect, 2)
        apply_text = font.render("Apply", True, WHITE)
        popup_surface.blit(apply_text, (apply_button_rect.centerx - apply_text.get_width() // 2,
                                       apply_button_rect.centery - apply_text.get_height() // 2))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                x -= popup_rect.left
                y -= popup_rect.top
                
                if close_button_rect.collidepoint(x, y):
                    running = False
                    return None
                
                elif apply_button_rect.collidepoint(x, y):
                    return observation_ratio
                
                if (slider_x <= x <= slider_x + slider_width and 
                    slider_y - 10 <= y <= slider_y + 20):
                    dragging_slider = True
                    observation_ratio = max(0, min(1, (x - slider_x) / slider_width))
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging_slider = False
                
            elif event.type == pygame.MOUSEMOTION and dragging_slider:
                x = event.pos[0] - popup_rect.left
                observation_ratio = max(0, min(1, (x - slider_x) / slider_width))
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    return None
                
                elif event.key == pygame.K_LEFT:
                    observation_ratio = max(0, observation_ratio - 0.05)
                elif event.key == pygame.K_RIGHT:
                    observation_ratio = min(1, observation_ratio + 0.05)
        
        screen.blit(popup_surface, popup_rect)
        pygame.display.flip()
        clock.tick(60)
    
    return None

"""Hàm main"""
def main():
    global running, solution, step, selected_algorithm, paused, start_time, elapsed_time
    running = False
    paused = False
    current_state = start_puzzle
    no_solution_message = None
    selected_tile = None

    def path_to_states(start_state, move_path):
        if not move_path:
            return None
            
        states = [start_state]
        current = [row[:] for row in start_state]
        
        for move in move_path:
            r, c = next((r, c) for r in range(3) for c in range(3) if current[r][c] == 0)
            nr, nc = move
            new_state = [row[:] for row in current]
            new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
            states.append(new_state)
            current = new_state
            
        return states

    algorithm_map = {
        # Uninformed Search Algorithms
        "BFS": bfs,
        "DFS": dfs,
        "UCS": ucs,
        "ID": interactive_deepening,
        
        # Informed Search Algorithms
        "Greedy": greedy,
        "A*": astar,
        "IDA*": ida_star,
        
        # Local Search Algorithms
        "Simple HC": simple_hill_climbing,
        "Steepest HC": steepest_ascent_hill_climbing,
        "Stochastic HC": stochastic_hill_climbing,
        "SA": simulated_annealing,
        "Beam": beam_search,
        "Genetic": genetic_algorithm,
        
        # Complex Environment Search
        "AND-OR": lambda start: path_to_states(start, and_or_tree_search(start, goal_puzzle)),
        "Belief State": lambda start: None,
        "PO": lambda start: None,
        
        # Constraint Satisfaction Problems
        "MC": lambda start: min_conflicts(start, goal_puzzle, use_random_start=True),
        "BACK": backtracking_search,
        "BACK-FC": backtracking_with_forward_checking,

        # Reinforcement Learning
        "Q-Learning": lambda start: path_to_states(start, q_learning_solve(start, goal_puzzle)[0])
    }

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos

                # Logic chọn 1 ô
                board_size = 3 * TILE_SIZE + 20
                board_x = (WIDTH - board_size) // 2
                board_y = HEIGHT // 2 - board_size // 2
                for row in range(3):
                    for col in range(3):
                        tile_x = board_x + col * TILE_SIZE + 10
                        tile_y = board_y + row * TILE_SIZE + 10
                        if tile_x <= x <= tile_x + TILE_SIZE - 10 and tile_y <= y <= tile_y + TILE_SIZE - 10:
                            if selected_tile == (row, col):
                                selected_tile = None
                            else:
                                selected_tile = (row, col)
                            break

                # Logic chọn thuật toán
                button_y_top = MARGIN + 80
                button_y_bottom = button_y_top + BUTTON_HEIGHT + MARGIN
                algo_per_row = 9
                total_buttons_width = algo_per_row * (BUTTON_WIDTH + MARGIN) - MARGIN
                start_x = (WIDTH - total_buttons_width) // 2

                for i, algo in enumerate(algorithms):
                    x_pos = start_x + (i % algo_per_row) * (BUTTON_WIDTH + MARGIN)
                    y_pos = button_y_top + (i // algo_per_row) * (BUTTON_HEIGHT + MARGIN)
                    if x_pos <= x <= x_pos + BUTTON_WIDTH and y_pos <= y <= y_pos + BUTTON_HEIGHT:
                        selected_algorithm = algo
                        break

                # Logic chọn ô điều khiển
                control_y = HEIGHT - 120
                total_control_width = 4 * (BUTTON_WIDTH + MARGIN) - MARGIN
                control_x = (WIDTH - total_control_width) // 2

                # Nút Start
                if control_x <= x <= control_x + BUTTON_WIDTH and control_y <= y <= control_y + BUTTON_HEIGHT:
                    running = True
                    paused = False
                    no_solution_message = None
                    
                    try:
                        # Bắt đầu tính giờ
                        start_time = time.time()
                        # Các thuật toán đặc biệt
                        if selected_algorithm == "Belief State":
                            selected_belief_state = show_belief_states_popup(goal_puzzle)
                            
                            if selected_belief_state is None or len(selected_belief_state) == 0:
                                no_solution_message = "No belief state selected"
                                running = False
                                continue
                                
                            belief_state = selected_belief_state[0]
                            
                            path = no_observation_search([belief_state], goal_puzzle)
                            solution = path_to_states(belief_state, path) if path else None
                                
                        elif selected_algorithm == "PO":
                            observation_ratio = show_po_settings_popup()
                            
                            if observation_ratio is None:
                                no_solution_message = "Possibilities not selected"
                                running = False
                                continue
                                
                            path = partially_observable_search(current_state, goal_puzzle, observation_ratio)
                            solution = path_to_states(current_state, path) if path else None

                        elif selected_algorithm == "Q-Learning":
                            path, algo_elapsed_time = q_learning_solve(current_state, goal_puzzle)
                            solution = path_to_states(current_state, path) if path else None
                            elapsed_time = algo_elapsed_time

                        # Thuật toán khác
                        else: 
                            solution = algorithm_map[selected_algorithm](current_state)
                            elapsed_time = time.time() - start_time
                        
                        if selected_algorithm != "Q-Learning":
                            elapsed_time = time.time() - start_time

                        # Thuật toán backtracking
                        if selected_algorithm == "BACK":
                            if not solution or len(solution) == 0:
                                no_solution_message = "No solution found!"
                                running = False
                        # Thuật toán thuộc nhóm complex environment search
                        elif solution is None or len(solution) == 0:
                            algo_desc = {
                                "Belief State": "Belief State",
                                "PO": " Partially Observable",
                                "AND-OR": " AND-OR",
                            }.get(selected_algorithm, "")
                            
                            no_solution_message = f"No solution found for {algo_desc}"
                            running = False
                                
                        step = 0
                        
                    except Exception as e:
                        print(f"Error in {selected_algorithm}: {str(e)}")
                        solution = None
                        no_solution_message = "An error occurred during search"
                        running = False
                        elapsed_time = time.time() - start_time

                # Nút Pause
                elif control_x + (BUTTON_WIDTH + MARGIN) <= x <= control_x + 2 * BUTTON_WIDTH + MARGIN and control_y <= y <= control_y + BUTTON_HEIGHT:
                    running = False
                    paused = True

                # Nút Reset
                elif control_x + 2 * (BUTTON_WIDTH + MARGIN) <= x <= control_x + 3 * BUTTON_WIDTH + 2 * MARGIN and control_y <= y <= control_y + BUTTON_HEIGHT:
                    running = False
                    paused = False
                    step = 0
                    solution = []
                    current_state = start_puzzle
                    elapsed_time = 0
                    start_time = 0
                    no_solution_message = None
                    
                # Nút steps
                elif control_x + 3 * (BUTTON_WIDTH + MARGIN) <= x <= control_x + 4 * BUTTON_WIDTH + 3 * MARGIN and control_y <= y <= control_y + BUTTON_HEIGHT:
                    if solution:
                        show_steps_window(solution)

            elif event.type == pygame.KEYDOWN and selected_tile:
                row, col = selected_tile
                if event.key == pygame.K_0:
                    current_state[row][col] = 0
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    value = event.key - pygame.K_0
                    if value not in sum(current_state, []):
                        current_state[row][col] = value

        # Vẽ animations và trạng thái
        if running:
            if solution and step < len(solution):
                current_state = solution[step]
                draw_puzzle(current_state, selected_tile)
                step += 1
                pygame.time.delay(300)
            else:
                running = False
        else:
            draw_puzzle(current_state, selected_tile)

        if no_solution_message:
            text = font.render(no_solution_message, True, DARK_BLUE)
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2))

        pygame.display.flip()
        clock.tick(60)
        
if __name__ == "__main__":
    main()