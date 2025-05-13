import matplotlib.pyplot as plt
import numpy as np
import time
import json
import os
from PhanVietTuan_23110355_BaoCaoCaNhan import * 

# Đặt style cho matplotlib
plt.style.use('seaborn-v0_8-colorblind')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

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
        
        # Complex Environment Search - Sử dụng hàm benchmark riêng
        "AND-OR": lambda s: run_andor_benchmark(s),
        "Belief State": lambda s: run_belief_state_benchmark(observed_state),
        "PO": lambda s: run_po_benchmark(observed_state, 0.5),
        
        # CSPs
        "MC": lambda s: min_conflicts(s, goal_puzzle, max_steps=1000, max_restarts=5),
        "BACK": backtracking_search,
        "BACK-FC": backtracking_with_forward_checking,
        
        # RL
        "Q-Learning": lambda s: path_to_states(s, q_learning_solve(s, goal_puzzle)[0])
    }
    
    # Cấu trúc kết quả mở rộng để lưu thêm thông tin về môi trường phức tạp
    results = {
        name: {
            "time": 0, 
            "path_length": 0, 
            "success": False,
            "nodes_expanded": 0  # Thêm số node mở rộng nếu có
        } 
        for name in algorithms.keys()
    }
    
    # Thêm cấu trúc đặc biệt cho các thuật toán môi trường phức tạp
    complex_results = {
        "Belief State": {
            "num_states": 0,
            "num_solved": 0,
            "success_rate": 0,
            "total_time": 0,
            "avg_time": 0,
            "avg_steps": 0
        },
        "PO": {
            "num_states": 0,
            "num_solved": 0,
            "success_rate": 0,
            "total_time": 0,
            "avg_time": 0,
            "avg_steps": 0
        },
        "AND-OR": {
            "time": 0,
            "steps": 0,
            "success": False
        }
    }
    
    print("Bắt đầu benchmark với trường hợp đơn giản...")
    
    # Chạy từng thuật toán
    for algo_name, algo_func in algorithms.items():
        print(f"Đang benchmark {algo_name}...")
        
        times = []
        path_lengths = []
        nodes_expanded = []
        success = False
        
        for run in range(num_runs):
            try:
                if algo_name in ["Belief State", "PO", "AND-OR"]:
                    # Các thuật toán đặc biệt sẽ trả về lời giải đầu tiên tìm được
                    # và cập nhật complex_results
                    solution, stats = algo_func(test_case)
                    
                    # Cập nhật thống kê đặc biệt
                    if algo_name in complex_results:
                        complex_results[algo_name].update(stats)
                else:
                    # Các thuật toán thông thường
                    start_time = time.time()
                    solution = algo_func(test_case)
                    end_time = time.time()
                    
                    if solution:
                        times.append(end_time - start_time)
                        path_lengths.append(len(solution))
                        # Nếu thuật toán trả về số node mở rộng (có thể thêm vào các hàm thuật toán)
                        # nodes_expanded.append(num_nodes)
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
            if nodes_expanded:
                results[algo_name]["nodes_expanded"] = np.mean(nodes_expanded)
    
    # Cập nhật kết quả cho các thuật toán phức tạp
    for algo_name in ["Belief State", "PO", "AND-OR"]:
        if complex_results[algo_name].get("success_rate", 0) > 0 or complex_results[algo_name].get("success", False):
            results[algo_name]["success"] = True
            if "avg_steps" in complex_results[algo_name] and complex_results[algo_name]["avg_steps"] > 0:
                results[algo_name]["path_length"] = complex_results[algo_name]["avg_steps"]
            if "steps" in complex_results[algo_name] and complex_results[algo_name]["steps"] > 0:
                results[algo_name]["path_length"] = complex_results[algo_name]["steps"]
            if "avg_time" in complex_results[algo_name]:
                results[algo_name]["time"] = complex_results[algo_name]["avg_time"]
            if "time" in complex_results[algo_name]:
                results[algo_name]["time"] = complex_results[algo_name]["time"]
    
    # Lưu kết quả dưới dạng JSON để sử dụng sau này
    with open("assets/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Lưu kết quả chi tiết cho thuật toán phức tạp
    with open("assets/complex_benchmark_results.json", "w") as f:
        json.dump(complex_results, f, indent=2)
    
    # Tạo biểu đồ
    create_charts(results)
    create_complex_comparison_chart(complex_results)
    
    return results, complex_results

def run_belief_state_benchmark(observed_state):
    """Hàm chạy benchmark đặc biệt cho Belief State"""
    print("\n=== BELIEF STATE BENCHMARK ===")
    
    # Thống kê
    belief_stats = {
        "num_states": 0,
        "num_solved": 0,
        "success_rate": 0,
        "total_time": 0,
        "avg_time": 0,
        "avg_steps": 0
    }
    
    start_time = time.time()
    
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
    
    # Cập nhật số lượng belief states
    belief_stats["num_states"] = len(belief_states)
    print(f"- Số belief states đã tạo: {len(belief_states)}")
    
    # Tìm kiếm lời giải cho mỗi belief state
    first_solution = None
    total_steps = 0
    
    for state in belief_states:
        solve_start_time = time.time()
        if is_solvable(state, goal_puzzle):
            # Dùng A* để tìm lời giải
            solution = astar(state)
            if solution:
                belief_stats["num_solved"] += 1
                total_steps += len(solution)
                if first_solution is None:
                    first_solution = solution
                    print(f"- Lời giải đầu tiên có độ dài: {len(solution)}")
        solve_end_time = time.time()
        belief_stats["total_time"] += (solve_end_time - solve_start_time)
    
    end_time = time.time()
    
    # Cập nhật thống kê
    belief_stats["success_rate"] = belief_stats["num_solved"] / max(1, belief_stats["num_states"])
    belief_stats["avg_time"] = belief_stats["total_time"] / max(1, belief_stats["num_states"])
    belief_stats["avg_steps"] = total_steps / max(1, belief_stats["num_solved"]) if belief_stats["num_solved"] > 0 else 0
    
    # In kết quả
    print(f"- Số trạng thái giải được: {belief_stats['num_solved']}")
    print(f"- Tỷ lệ thành công: {belief_stats['success_rate']:.2%}")
    print(f"- Tổng thời gian: {belief_stats['total_time']:.4f}s")
    print(f"- Thời gian trung bình/trạng thái: {belief_stats['avg_time']:.4f}s")
    if belief_stats["num_solved"] > 0:
        print(f"- Số bước trung bình: {belief_stats['avg_steps']:.2f}")
    
    # Tạo biểu đồ riêng cho Belief State
    create_belief_state_chart(belief_stats)
    
    # Trả về lời giải đầu tiên tìm được (nếu có) cho mục đích benchmark chung
    return first_solution, belief_stats

def run_po_benchmark(observed_state, observation_ratio=0.5):
    """Hàm chạy benchmark đặc biệt cho Partially Observable"""
    print("\n=== PARTIALLY OBSERVABLE BENCHMARK ===")
    
    # Thống kê
    po_stats = {
        "num_states": 0,
        "num_solved": 0,
        "success_rate": 0,
        "total_time": 0,
        "avg_time": 0,
        "avg_steps": 0
    }
    
    start_time = time.time()
    
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
    
    # Cập nhật số lượng trạng thái
    po_stats["num_states"] = len(possible_states)
    print(f"- Số trạng thái khả dĩ đã tạo: {len(possible_states)}")
    
    # Tìm kiếm giải pháp cho mỗi trạng thái khả thi
    first_solution = None
    total_steps = 0
    
    for state in possible_states:
        solve_start_time = time.time()
        if is_solvable(state, goal_puzzle):
            # Dùng A* để tìm lời giải
            solution = astar(state)
            if solution:
                po_stats["num_solved"] += 1
                total_steps += len(solution)
                if first_solution is None:
                    first_solution = solution
                    print(f"- Lời giải đầu tiên có độ dài: {len(solution)}")
        solve_end_time = time.time()
        po_stats["total_time"] += (solve_end_time - solve_start_time)
    
    end_time = time.time()
    
    # Cập nhật thống kê
    po_stats["success_rate"] = po_stats["num_solved"] / max(1, po_stats["num_states"])
    po_stats["avg_time"] = po_stats["total_time"] / max(1, po_stats["num_states"])
    po_stats["avg_steps"] = total_steps / max(1, po_stats["num_solved"]) if po_stats["num_solved"] > 0 else 0
    
    # In kết quả
    print(f"- Số trạng thái giải được: {po_stats['num_solved']}")
    print(f"- Tỷ lệ thành công: {po_stats['success_rate']:.2%}")
    print(f"- Tổng thời gian: {po_stats['total_time']:.4f}s")
    print(f"- Thời gian trung bình/trạng thái: {po_stats['avg_time']:.4f}s")
    if po_stats["num_solved"] > 0:
        print(f"- Số bước trung bình: {po_stats['avg_steps']:.2f}")
    
    # Tạo biểu đồ riêng cho PO
    create_po_chart(po_stats)
    
    # Trả về lời giải đầu tiên tìm được (nếu có) cho mục đích benchmark chung
    return first_solution, po_stats

def run_andor_benchmark(start):
    """Hàm chạy benchmark đặc biệt cho AND-OR Search"""
    print("\n=== AND-OR SEARCH BENCHMARK ===")
    
    start_time = time.time()
    path = and_or_tree_search(start, goal_puzzle)
    end_time = time.time()
    
    elapsed = end_time - start_time
    solution = path_to_states(start, path) if path else None
    
    # Thống kê
    andor_stats = {
        "time": elapsed,
        "steps": len(solution) if solution else 0,
        "success": solution is not None
    }
    
    # In kết quả
    print(f"- Thời gian thực thi: {elapsed:.4f}s")
    if solution:
        print(f"- Số bước đi: {len(solution)}")
        print(f"- Tìm được lời giải: Có")
    else:
        print(f"- Tìm được lời giải: Không")
    
    # Tạo biểu đồ riêng cho AND-OR
    create_andor_chart(andor_stats)
    
    return solution, andor_stats

def create_charts(results):
    algorithm_groups = {
        "uninformed": ["BFS", "DFS", "UCS", "ID"],
        "informed": ["Greedy", "A*", "IDA*"],
        "local": ["Simple HC", "Steepest HC", "Stochastic HC", "SA", "Beam", "Genetic"],
        "complex": ["AND-OR", "Belief State", "PO"],
        "csp": ["MC", "BACK", "BACK-FC"],
        "rl": ["Q-Learning"]
    }
    group_titles = {
        "uninformed": "Thuật toán tìm kiếm không thông tin",
        "informed": "Thuật toán tìm kiếm có thông tin",
        "local": "Thuật toán tìm kiếm cục bộ",
        "complex": "Thuật toán tìm kiếm trong môi trường phức tạp",
        "csp": "Thuật toán tìm kiếm trong điều kiện ràng buộc",
        "rl": "Thuật toán học tăng cường"
    }
    for group_name, group_algos in algorithm_groups.items():
        filtered_algos = [algo for algo in group_algos if algo in results and results[algo]["success"]]
        if not filtered_algos:
            continue
        plt.close('all')
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(111)
        x = np.arange(len(filtered_algos))
        width = 0.35
        g_times = [results[algo]["time"] for algo in filtered_algos]
        g_paths = [results[algo]["path_length"] for algo in filtered_algos]
        time_colors = plt.cm.Blues(np.linspace(0.6, 0.9, len(filtered_algos)))
        path_colors = plt.cm.Greens(np.linspace(0.6, 0.9, len(filtered_algos)))
        bar1 = ax1.bar(x - width/2, g_times, width, color=time_colors, label='Thời gian (giây)',
                      edgecolor='black', linewidth=0.5, alpha=0.8)
        ax1.set_ylabel('Thời gian (giây)', color='#1f77b4', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        for i, v in enumerate(g_times):
            if v > 0:
                ax1.text(i - width/2, v + 0.01, f'{v:.2f}s', color='navy', fontweight='bold', ha='center', va='bottom')
        ax2 = ax1.twinx()
        bar2 = ax2.bar(x + width/2, g_paths, width, color=path_colors, label='Độ dài đường đi',
                      edgecolor='black', linewidth=0.5, alpha=0.8)
        ax2.set_ylabel('Độ dài đường đi', color='#2ca02c', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#2ca02c')
        for i, v in enumerate(g_paths):
            if v > 0:
                ax2.text(i + width/2, v + 0.5, f'{int(v)}', color='darkgreen', fontweight='bold', ha='center', va='bottom')
        ax1.set_title(f'{group_titles.get(group_name, group_name.title())}', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(filtered_algos, rotation=45, ha='right')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'assets/{group_name}_compare.png', dpi=150)
        print(f"Đã lưu biểu đồ so sánh nhóm {group_name}")

def create_complex_comparison_chart(complex_results):
    if not complex_results:
        return
    plt.close('all')
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    labels = ["Belief State", "PO", "AND-OR"]
    metrics = {
        "Tỷ lệ thành công": [
            complex_results["Belief State"]["success_rate"],
            complex_results["PO"]["success_rate"],
            complex_results["AND-OR"]["success"] * 1.0
        ],
        "Thời gian": [
            complex_results["Belief State"]["avg_time"],
            complex_results["PO"]["avg_time"],
            complex_results["AND-OR"]["time"]
        ],
        "Số bước": [
            complex_results["Belief State"]["avg_steps"],
            complex_results["PO"]["avg_steps"],
            complex_results["AND-OR"]["steps"]
        ]
    }
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    axs[0].bar(labels, metrics["Tỷ lệ thành công"], color=colors)
    axs[0].set_ylabel('Tỷ lệ')
    axs[0].set_title('Tỷ lệ thành công', fontweight='bold')
    for i, v in enumerate(metrics["Tỷ lệ thành công"]):
        axs[0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    axs[1].bar(labels, metrics["Thời gian"], color=colors)
    axs[1].set_ylabel('Giây')
    axs[1].set_title('Thời gian trung bình', fontweight='bold')
    for i, v in enumerate(metrics["Thời gian"]):
        axs[1].text(i, v + 0.02, f'{v:.3f}s', ha='center', fontweight='bold')
    axs[2].bar(labels, metrics["Số bước"], color=colors)
    axs[2].set_ylabel('Số bước')
    axs[2].set_title('Số bước trung bình', fontweight='bold')
    for i, v in enumerate(metrics["Số bước"]):
        axs[2].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')
    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.3)
    fig.suptitle('So sánh hiệu suất các thuật toán trong môi trường phức tạp', fontweight='bold', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig('assets/complex_compare.png', dpi=150)
    print("Đã lưu biểu đồ so sánh môi trường phức tạp")

def create_summary_chart(results):
    """Tạo biểu đồ tổng hợp so sánh tất cả các thuật toán thành công"""
    # Lọc thuật toán thành công
    successful_algos = [algo for algo in results.keys() if results[algo]["success"]]
    
    if len(successful_algos) < 2:
        return
    
    # Đóng tất cả các figure để tránh xung đột
    plt.close('all')

    # Tạo figure và subplots - CHỈ TẠO MỘT LẦN
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Dữ liệu
    times = [results[algo]["time"] for algo in successful_algos]
    path_lengths = [results[algo]["path_length"] for algo in successful_algos]
    
    # Sắp xếp thuật toán theo thời gian
    sorted_indices = np.argsort(times)
    sorted_algos = [successful_algos[i] for i in sorted_indices]
    sorted_times = [times[i] for i in sorted_indices]
    sorted_paths = [path_lengths[i] for i in sorted_indices]
    
    # Tạo vị trí x
    x = np.arange(len(sorted_algos))
    width = 0.35
    
    # Sử dụng color gradient cho các cột
    time_colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(sorted_algos)))
    path_colors = plt.cm.Greens(np.linspace(0.5, 0.9, len(sorted_algos)))
    
    # Biểu đồ thời gian
    time_bars = ax1.bar(x, sorted_times, width*1.5, color=time_colors, 
                       edgecolor='black', linewidth=0.5, alpha=0.8)
    ax1.set_ylabel('Thời gian (giây)', fontweight='bold')
    ax1.set_title('So sánh thời gian thực thi', fontweight='bold')
    
    # Thêm nhãn
    for i, v in enumerate(sorted_times):
        ax1.text(i, v + 0.01, f'{v:.3f}s', 
                color='navy', fontweight='bold', ha='center', va='bottom')
    
    # Biểu đồ độ dài đường đi
    path_bars = ax2.bar(x, sorted_paths, width*1.5, color=path_colors,
                       edgecolor='black', linewidth=0.5, alpha=0.8)
    ax2.set_ylabel('Độ dài đường đi', fontweight='bold')
    ax2.set_title('So sánh độ dài đường đi', fontweight='bold')
    
    # Thêm nhãn
    for i, v in enumerate(sorted_paths):
        ax2.text(i, v + 0.5, f'{int(v)}', 
                color='darkgreen', fontweight='bold', ha='center', va='bottom')
    
    # Thiết lập trục x
    ax2.set_xticks(x)
    ax2.set_xticklabels(sorted_algos, rotation=45, ha='right')
    
    # Thêm lưới
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    fig.suptitle('So sánh tổng hợp tất cả thuật toán', fontsize=16, fontweight='bold', y=0.98)
    fig.subplots_adjust(bottom=0.2, hspace=0.3, top=0.9)
    
    fig.savefig(f'assets/overall_comparison.png', dpi=300, bbox_inches='tight')
    print("Đã lưu biểu đồ tổng hợp")


def create_belief_state_chart(stats):
    """Tạo biểu đồ chi tiết cho Belief State"""
    # Đóng tất cả các figure trước
    plt.close('all')
    
    fig = plt.figure(figsize=(10, 5))
    
    # Dữ liệu
    labels = ['Số trạng thái', 'Số giải được', 'Thời gian TB (s)', 'Số bước TB']
    values = [stats["num_states"], stats["num_solved"], stats["avg_time"], stats["avg_steps"]]
    
    # Màu sắc đẹp
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    # Vẽ biểu đồ
    ax = fig.add_subplot(111)
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Thêm nhãn giá trị
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
        
    ax.set_title('Belief State Search - Hiệu suất', fontweight='bold')
    ax.set_ylabel('Giá trị')
    
    # Thêm thông tin tỷ lệ thành công
    fig.text(0.5, 0.02, f'Tỷ lệ thành công: {stats["success_rate"]:.2%}', 
             ha="center", fontsize=12, fontweight='bold',
             bbox={"facecolor":"#f1c40f", "edgecolor":"#e67e22", "alpha":0.2, "pad":5})
    
    # Thêm lưới
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Sử dụng subplots_adjust
    fig.subplots_adjust(bottom=0.15)
    
    fig.savefig('assets/belief_state_benchmark.png', dpi=300, bbox_inches='tight')
    print("Đã lưu biểu đồ Belief State")

def create_po_chart(stats):
    """Tạo biểu đồ chi tiết cho Partially Observable"""
    # Đóng tất cả các figure trước
    plt.close('all')
    
    fig = plt.figure(figsize=(10, 5))
    
    # Dữ liệu
    labels = ['Số trạng thái', 'Số giải được', 'Thời gian TB (s)', 'Số bước TB']
    values = [stats["num_states"], stats["num_solved"], stats["avg_time"], stats["avg_steps"]]
    
    # Màu sắc đẹp
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    # Vẽ biểu đồ
    ax = fig.add_subplot(111)
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Thêm nhãn giá trị
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
        
    ax.set_title('Partially Observable Search - Hiệu suất', fontweight='bold')
    ax.set_ylabel('Giá trị')
    
    # Thêm thông tin tỷ lệ thành công
    fig.text(0.5, 0.02, f'Tỷ lệ thành công: {stats["success_rate"]:.2%}', 
             ha="center", fontsize=12, fontweight='bold',
             bbox={"facecolor":"#f1c40f", "edgecolor":"#e67e22", "alpha":0.2, "pad":5})
    
    # Thêm lưới
    ax.grid(True, linestyle='--', alpha=0.3)
    
    fig.subplots_adjust(bottom=0.15)
    
    fig.savefig('assets/po_benchmark.png', dpi=300, bbox_inches='tight')
    print("Đã lưu biểu đồ PO")

def create_andor_chart(stats):
    """Tạo biểu đồ chi tiết cho AND-OR"""
    # Đóng tất cả các figure trước
    plt.close('all')
    
    fig = plt.figure(figsize=(8, 5))
    
    # Dữ liệu
    labels = ['Thời gian (s)', 'Số bước']
    values = [stats["time"], stats["steps"]]
    
    # Màu sắc
    colors = ['#3498db', '#2ecc71']
    
    # Vẽ biểu đồ
    ax = fig.add_subplot(111)
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Thêm nhãn giá trị
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
        
    ax.set_title('AND-OR Search - Hiệu suất', fontweight='bold')
    ax.set_ylabel('Giá trị')
    
    # Thêm thông tin thành công
    status = "Thành công" if stats["success"] else "Không thành công"
    fig.text(0.5, 0.02, f'Kết quả: {status}', 
             ha="center", fontsize=12, fontweight='bold',
             bbox={"facecolor":"#f1c40f", "edgecolor":"#e67e22", "alpha":0.2, "pad":5})
    
    # Thêm lưới
    ax.grid(True, linestyle='--', alpha=0.3)
    
    fig.subplots_adjust(bottom=0.15)
    
    fig.savefig('assets/andor_benchmark.png', dpi=300, bbox_inches='tight')
    print("Đã lưu biểu đồ AND-OR")

if __name__ == "__main__":
    benchmark_algorithms(3)  # Chạy mỗi thuật toán 3 lần
    print("Benchmark hoàn tất. Kết quả lưu trong thư mục assets/")