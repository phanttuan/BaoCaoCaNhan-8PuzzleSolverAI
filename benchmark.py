import matplotlib.pyplot as plt
import numpy as np
import time
import json
import os
from PhanVietTuan_23110355_BaoCaoCaNhan import * 

os.makedirs("assets", exist_ok=True)

def benchmark_algorithms(num_runs=3):
    """
    Đo và so sánh hiệu suất các thuật toán
    """
    # Chỉ dùng một test case đơn giản
    test_case = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]  # Trường hợp đơn giản
    
    # Tạo trạng thái đơn giản cho Belief State và PO
    def create_simple_belief_state():
        state = [["?" if (i+j) % 3 == 0 else test_case[i][j] for j in range(3)] for i in range(3)]
        return state, test_case
    
    observed_state, true_state = create_simple_belief_state()
    
    # Danh sách các thuật toán cần benchmark
    algorithms = {
        # Uninformed Search
        "BFS": bfs,
        "DFS": dfs,
        "UCS": ucs,
        "ID": interactive_deepening,
        
        # Informed Search
        "Greedy": greedy,
        "A*": astar,
        "IDA*": ida_star,
        
        # Local Search
        "Simple HC": simple_hill_climbing,
        "Steepest HC": steepest_ascent_hill_climbing,
        "Stochastic HC": stochastic_hill_climbing,
        "SA": simulated_annealing,
        "Beam": beam_search,
        "Genetic": genetic_algorithm,
        
        # Complex Environment Search
        "AND-OR": lambda s: path_to_states(s, and_or_tree_search(s, goal_puzzle)),
        
        # Belief State - Benchmark phiên bản đơn giản
        "Belief State": lambda s: run_belief_state_benchmark(observed_state),
        
        # PO - Benchmark phiên bản đơn giản
        "PO": lambda s: run_po_benchmark(observed_state, 0.5),
        
        # CSPs
        "MC": lambda s: min_conflicts(s, goal_puzzle, max_steps=1000, max_restarts=5),
        "BACK": backtracking_search,
        "BACK-FC": backtracking_with_forward_checking,
        
        # RL
        "Q-Learning": lambda s: path_to_states(s, q_learning_solve(s, goal_puzzle)[0])
    }
    
    # Cấu trúc kết quả
    results = {name: {"time": 0, "path_length": 0, "success": False} for name in algorithms.keys()}
    
    print("Bắt đầu benchmark với trường hợp đơn giản...")
    
    # Chạy từng thuật toán
    for algo_name, algo_func in algorithms.items():
        print(f"Đang benchmark {algo_name}...")
        
        times = []
        path_lengths = []
        success = False
        
        for run in range(num_runs):
            try:
                start_time = time.time()
                
                # Các thuật toán đặc biệt
                if algo_name in ["Belief State", "PO"]:
                    solution = algo_func(None)  # Không cần tham số input
                else:
                    solution = algo_func(test_case)
                    
                end_time = time.time()
                
                if solution:
                    times.append(end_time - start_time)
                    path_lengths.append(len(solution))
                    success = True
                    print(f"  - {algo_name} lần {run+1}: {end_time - start_time:.3f}s, độ dài: {len(solution)}")
                else:
                    print(f"  - {algo_name} lần {run+1}: Không tìm được lời giải")
            except Exception as e:
                print(f"  - {algo_name} lỗi: {str(e)}")
        
        if success:
            results[algo_name]["time"] = np.mean(times)
            results[algo_name]["path_length"] = np.mean(path_lengths)
            results[algo_name]["success"] = True
    
    # Lưu kết quả dưới dạng JSON để sử dụng sau này
    with open("assets/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Tạo biểu đồ
    create_charts(results)
    
    return results

def run_belief_state_benchmark(observed_state):
    """Hàm chạy benchmark đặc biệt cho Belief State"""
    # Giới hạn số lượng belief states để tránh tiêu tốn quá nhiều thời gian
    max_belief_states = 10
    
    # Tạo một số belief states có giới hạn
    belief_states = []
    
    # Tìm vị trí các ô chưa biết
    unknown_positions = []
    for r in range(3):
        for c in range(3):
            if observed_state[r][c] == "?":
                unknown_positions.append((r, c))
    
    # Tạo một số cấu hình ngẫu nhiên cho các vị trí không biết
    missing_values = [v for v in range(9) if all(v != observed_state[r][c] 
                                             for r in range(3) for c in range(3) 
                                             if observed_state[r][c] != "?")]
    
    # Tạo tối đa 10 belief states
    import random
    from itertools import permutations
    
    all_permutations = list(permutations(missing_values))
    if len(all_permutations) > max_belief_states:
        selected_permutations = random.sample(all_permutations, max_belief_states)
    else:
        selected_permutations = all_permutations
    
    for perm in selected_permutations:
        new_state = [row[:] for row in observed_state]
        for i, (r, c) in enumerate(unknown_positions):
            if i < len(perm):
                new_state[r][c] = perm[i]
        
        # Chuyển các ô "?" còn lại thành 0 (nếu có)
        for r in range(3):
            for c in range(3):
                if new_state[r][c] == "?":
                    new_state[r][c] = 0
                    
        belief_states.append(new_state)
    
    # Tìm kiếm lời giải cho mỗi belief state
    for state in belief_states:
        if is_solvable(state, goal_puzzle):
            # Dùng A* để tìm lời giải
            solution = astar(state)
            if solution:
                return solution
    
    # Nếu không tìm thấy lời giải cho bất kỳ belief state nào
    return None

def run_po_benchmark(observed_state, observation_ratio=0.5):
    """Hàm chạy benchmark đặc biệt cho Partially Observable"""
    # Giới hạn số lượng trạng thái có thể sinh ra
    max_states = 10
    
    # Tạo một trạng thái quan sát được đơn giản
    flat_state = [observed_state[r][c] for r in range(3) for c in range(3)]
    
    # Xác định số lượng ô có thể quan sát được
    num_observable = max(3, int(9 * observation_ratio))
    
    # Các vị trí có thể quan sát được
    unknown_positions = []
    for r in range(3):
        for c in range(3):
            if observed_state[r][c] == "?":
                unknown_positions.append((r, c))
    
    # Các giá trị đã biết
    known_values = []
    for r in range(3):
        for c in range(3):
            if observed_state[r][c] != "?":
                known_values.append(observed_state[r][c])
    
    # Các giá trị chưa biết
    unknown_values = [i for i in range(9) if i not in known_values]
    
    # Tạo các trạng thái khả thi
    import random
    from itertools import permutations
    
    all_permutations = list(permutations(unknown_values))
    if len(all_permutations) > max_states:
        selected_permutations = random.sample(all_permutations, max_states)
    else:
        selected_permutations = all_permutations
    
    possible_states = []
    for perm in selected_permutations:
        new_state = [row[:] for row in observed_state]
        for i, (r, c) in enumerate(unknown_positions):
            if i < len(perm):
                new_state[r][c] = perm[i]
        
        possible_states.append(new_state)
    
    # Tìm kiếm giải pháp cho mỗi trạng thái khả thi
    for state in possible_states:
        if is_solvable(state, goal_puzzle):
            # Dùng A* để tìm lời giải
            solution = astar(state)
            if solution:
                return solution
    
    return None

def create_charts(results):
    """Tạo các biểu đồ so sánh từ kết quả benchmark"""
    
    # Đảm bảo tất cả thuật toán đều hiển thị trên biểu đồ
    all_algos = list(results.keys())
    
    # Lọc các thuật toán thành công hoặc cần hiển thị dù không thành công
    valid_algos = []
    for algo in all_algos:
        if results[algo]["success"] or algo in ["MC", "Belief State", "PO"]:  # Hiển thị MC, Belief State, PO ngay cả khi không thành công
            valid_algos.append(algo)
    
    if not valid_algos:
        print("Không có dữ liệu hiệu suất nào để tạo biểu đồ.")
        return
    
    # Dữ liệu cần vẽ
    times = [results[algo]["time"] for algo in valid_algos]
    path_lengths = [results[algo]["path_length"] for algo in valid_algos]
    
    # Tạo biểu đồ riêng cho từng nhóm thuật toán
    algorithm_groups = {
        "uninformed": ["BFS", "DFS", "UCS", "ID"],
        "informed": ["Greedy", "A*", "IDA*"],
        "local": ["Simple HC", "Steepest HC", "Stochastic HC", "SA", "Beam", "Genetic"],
        "complex": ["AND-OR", "Belief State", "PO"],
        "csp": ["MC", "BACK", "BACK-FC"],
        "rl": ["Q-Learning"]
    }
    
    # Vẽ biểu đồ cho từng nhóm
    for group_name, group_algos in algorithm_groups.items():
        # Lọc các thuật toán trong nhóm
        filtered_algos = []
        for algo in group_algos:
            if algo in valid_algos:
                filtered_algos.append(algo)
        
        if not filtered_algos:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Dữ liệu cho nhóm hiện tại
        g_times = []
        g_paths = []
        for algo in filtered_algos:
            # Nếu thuật toán không thành công, sử dụng giá trị 0
            if algo in results and results[algo]["success"]:
                g_times.append(results[algo]["time"])
                g_paths.append(results[algo]["path_length"])
            else:
                g_times.append(0)
                g_paths.append(0)
        
        # Tạo vị trí x cho các cột
        x = np.arange(len(filtered_algos))
        width = 0.35
        
        # Vẽ hai loại giá trị
        ax1 = plt.subplot(111)
        bar1 = ax1.bar(x - width/2, g_times, width, color='skyblue', label='Thời gian (giây)')
        ax1.set_ylabel('Thời gian (giây)', color='royalblue')
        ax1.tick_params(axis='y', labelcolor='royalblue')
        
        # Thêm nhãn thời gian
        for i, v in enumerate(g_times):
            ax1.text(i - width/2, v + 0.01, f'{v:.2f}s', 
                    color='navy', fontweight='bold', ha='center', va='bottom')
        
        # Trục thứ hai cho độ dài đường đi
        ax2 = ax1.twinx()
        bar2 = ax2.bar(x + width/2, g_paths, width, color='lightgreen', label='Độ dài đường đi')
        ax2.set_ylabel('Độ dài đường đi', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Thêm nhãn độ dài
        for i, v in enumerate(g_paths):
            ax2.text(i + width/2, v + 0.5, f'{int(v)}', 
                    color='darkgreen', fontweight='bold', ha='center', va='bottom')
        
        # Thiết lập nhãn trục x
        plt.title(f'So sánh hiệu suất nhóm {group_name.title()}')
        plt.xticks(x, filtered_algos, rotation=45, ha='right')
        
        # Thêm chú thích
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'assets/{group_name}_compare.png', dpi=300)
        print(f"Đã lưu biểu đồ so sánh nhóm {group_name}")

if __name__ == "__main__":
    benchmark_algorithms(3)  # Chạy mỗi thuật toán 3 lần
    print("Benchmark hoàn tất. Kết quả lưu trong thư mục assets/")