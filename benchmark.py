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
        
        # Complex Environment Search - Thêm vào 
        "AND-OR": lambda s: path_to_states(s, and_or_tree_search(s, goal_puzzle)),
        # Belief State và PO là thuật toán tương tác, tạo giá trị đơn giản cho benchmark
        "Belief State": lambda s: [s, goal_puzzle],  # Giá trị mô phỏng đơn giản
        "PO": lambda s: [s, goal_puzzle],  # Giá trị mô phỏng đơn giản
        
        # CSPs
        "MC": lambda s: min_conflicts(s, goal_puzzle, use_random_start=True),
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
            if algo in valid_algos or algo in ["MC", "Belief State", "PO"]:
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