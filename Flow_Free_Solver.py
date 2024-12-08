import numpy as np
import random
from collections import deque
import copy
import matplotlib.pyplot as plt

# 定義網格大小（例如5x5）
GRID_SIZE = 10
NUM_COLORS = 9  # 假設我們有五對顏色點

# 初始化參數
POPULATION_SIZE = 1000
GENERATIONS = 1000
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 5

# 初始設定的顏色
COLORS = ["R", "G", "B", "Y", "O","P","p","b","L"]  # 紅、綠、藍、黃、橙
# COLORS = ["R", "G", "B", "Y", "O"]  # 紅、綠、藍、黃、橙
# COLORS = ["R", "G", "B", "Y", "O","P"]  # 紅、綠、藍、黃、橙
color_path_lengths = {}  # 每個顏色的最短路徑長度
COLOR_MAP = {
    "R": "red",
    "G": "green",
    "B": "blue",
    "Y": "yellow",
    "O": "orange",
    "P": "purple",
    "p": "pink",
    "b": "brown",
    "L": "lightblue",
    "": "white" , # 空白格子
    "1": "black"  # 保護點
}
# COLOR_MAP = {
#     "R": "red",
#     "G": "green",
#     "B": "blue",
#     "Y": "yellow",
#     "O": "orange",
#     "": "white" , # 空白格子
#     "1": "black"  # 保護點
# }
# 隨機生成起點和終點
# endpoints = {
#     "R": [(1, 6), (5, 4)],  # Yellow
#     "B": [(0, 6), (6, 5)],  # Blue
#     "G": [(3, 3), (4, 2)],  # Green
#     "Y": [(4, 4), (5, 5)],  # Red
#     "O": [(1, 5), (2, 1)],  # Orange
#     "P": [(3, 4), (6, 6)]   # purple
# }

# endpoints = {
#     "Y": [(0, 4), (4, 2)],
#     "B": [(0, 5), (3, 3)],
#     "R": [(4, 3), (6, 1)],
#     "G": [(0, 6), (2, 2)],
#     "O": [(4, 5), (6, 0)],
# }
endpoints = {
    "R": [(6, 2), (9, 9)],  # Yellow
    "B": [(4, 8), (8, 9)],  # Blue
    "G": [(8, 1), (4, 5)],  # Green
    "Y": [(3, 2), (2, 7)],  # Red
    "O": [(4, 1), (8, 3)],  # Orange
    "P": [(6, 3), (4, 6)],   # purple
    "p": [(6, 4), (5, 5)],  # pink
    "b": [(9, 1), (7, 9)],  # brown
    "L": [(3, 1), (3, 8)]  # lightblue
}

fix = {}
# 定義個體
class Individual:
    def __init__(self):
        # self.genotype = np.random.choice(COLORS, (GRID_SIZE, GRID_SIZE))
        # 生成一個空白網格
        self.genotype = np.full((GRID_SIZE, GRID_SIZE), "", dtype=str)
        self.fitness = -1
        self.final = NUM_COLORS
        self.finished = False

    def calculate_fitness(self):
        color_connected = []
        cluster_num = self.compute_cluster(self.genotype)
        
        if cluster_num > self.final:
            self.fitness += (1/(cluster_num - self.final))* 1
        elif cluster_num == self.final:
            self.finished = True
            return
        
        # 檢查每個顏色的起點與終點是否連通
        path_score = 0
        connection_penalty = 0  # 未連線的懲罰
        cnt = 0
        for color, (start, end) in endpoints.items():
            if self.is_connected(color, start, end):
                path_score += 10
                color_connected.append(color)
                cnt += 1
            else:
                connection_penalty += 5  # 每個未連線的顏色降低 5 分
        if cnt == NUM_COLORS:
            self.finished = True
        # 假設適應度為覆蓋得分和路徑連通性得分的和
        self.fitness += path_score - connection_penalty


        # 計算每個顏色的數量
        color_counts = {color: 0 for color in COLORS}
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                color_counts[self.genotype[i, j]] += 1

        # 確保每種顏色至少有最短路徑長度的格子數 如果數量不足fitness減少
        for color in COLORS:
            required_length = color_path_lengths[color]
            actual_count = color_counts[color]
            if color in color_connected:
                if actual_count > required_length:
                    self.fitness -= (actual_count - required_length) * 1 # 調整權重可根據實驗結果進行調整
            else:
                if actual_count < required_length:
                    # 如果顏色數量不足，降低適應度
                    self.fitness -= (required_length - actual_count) * 5 # 調整權重可根據實驗結果進行調整
            
    
        
        self.fitness = max(0, self.fitness)  # 適應度不為負數


    def compute_cluster(self, genotype):
        rows, cols = genotype.shape
        visited = np.zeros((rows, cols), dtype=bool)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        cluster_count = 0

        def dfs(x, y, color):
            stack = [(x, y)]
            visited[x, y] = True
            while stack:
                cx, cy = stack.pop()
                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and genotype[nx, ny] == color:
                        visited[nx, ny] = True
                        stack.append((nx, ny))

        # 計算每個顏色的群集數量
        for i in range(rows):
            for j in range(cols):
                if not visited[i, j]:  # 如果尚未被訪問，則啟動新的群集計算
                    cluster_count += 1
                    dfs(i, j, genotype[i, j])

        return cluster_count



    def is_connected(self, color, start, end):
        # 這裡可以用深度優先搜索（DFS）或廣度優先搜索（BFS）來檢查路徑是否連通
        visited = set()
        stack = [start]
        
        while stack:
            x, y = stack.pop()
            if (x, y) == end:
                return True
            if (x, y) not in visited:
                visited.add((x, y))
                # 檢查四周相鄰的格子
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        if self.genotype[nx, ny] == color:
                            stack.append((nx, ny))
        
        return False


def enforce_fixed_points(individual):
    for color, (start, end) in endpoints.items():
        start_x, start_y = start
        end_x, end_y = end
        individual.genotype[start_x, start_y] = color
        individual.genotype[end_x, end_y] = color
    for color, points in fix.items():
        for point in points:
            x, y = point
            individual.genotype[x, y] = color
    


def calculate_shortest_path(start, end):
    queue = deque([(start, 0)])  # (位置, 路徑長度)
    visited = set([start])

    while queue:
        (x, y), length = queue.popleft()
        if (x, y) == end:
            return length-1  # 返回最短路徑長度

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 上下左右移動
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in visited:
                queue.append(((nx, ny), length + 1))
                visited.add((nx, ny))

    return float('inf')  # 若無法連接，返回無窮大               

def visualize_solution(genotype):
    plt.figure(figsize=(GRID_SIZE+1, GRID_SIZE+1))
    ax = plt.gca()
    
    # 遍歷每個格子，填充顏色
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color = genotype[i, j]
            # 使用 Rectangle 繪製網格，並按照左上角為 (0, 0) 進行顯示
            rect = plt.Rectangle((j, i), 1, 1, facecolor=COLOR_MAP[color], edgecolor='black')
            ax.add_patch(rect)

            # 標記起點和終點
            for c, (start, end) in endpoints.items():
                if (i, j) == start or (i, j) == end:
                    plt.text(j + 0.5, i + 0.5, c, color='black', ha='center', va='center', fontsize=12, fontweight='bold')

    # 設定網格和顯示屬性
    plt.xlim(0, GRID_SIZE)
    plt.ylim(0, GRID_SIZE)
    plt.xticks(np.arange(0, GRID_SIZE + 1, 1))
    plt.yticks(np.arange(0, GRID_SIZE + 1, 1))
    plt.grid(True)
    plt.gca().invert_yaxis()  # 反轉 y 軸，將 (0, 0) 設置在左上角
    plt.show(block=False)  # 非阻塞模式显示图片
    plt.pause(5)  # 暫停一段時間，以便觀察
    plt.close()

# 初始族群生成
def create_population(size):
    # 計算每個顏色至少需要的格子數（最短路徑長度）
    for color, (start, end) in endpoints.items():
        color_path_lengths[color] = calculate_shortest_path(start, end)
    
    print("每個顏色的最短路徑長度:", color_path_lengths)  # 確認結果
    

    # 初始化族群
    population = []
    for _ in range(size):
        individual = Individual()

        # 先將每個顏色的起點和終點填入網格
        for color, (start, end) in endpoints.items():
            start_x, start_y = start
            end_x, end_y = end
            individual.genotype[start_x, start_y] = color  # 起點
            individual.genotype[end_x, end_y] = color      # 終點
        
        find_initial_fix(individual)
        

        unconnected_colors = []
        for color, (start, end) in endpoints.items():
            start_x, start_y = start
            end_x, end_y = end

            # 檢查起點和終點是否已連接
            visited = set()
            stack = [start]
            connected = False
            while stack:
                current_x, current_y = stack.pop()
                if (current_x, current_y) == (end_x, end_y):
                    connected = True
                    break
                if (current_x, current_y) in visited:
                    continue
                visited.add((current_x, current_y))

                # 搜索相鄰格子
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = current_x + dx, current_y + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and individual.genotype[nx, ny] == color:
                        stack.append((nx, ny))

            if not connected:
                unconnected_colors.append(color)
        
        # 隨機選擇一個未連接的顏色
        selected_color = None
        if unconnected_colors:
            for i in range(len(unconnected_colors)-1):
              selected_color = random.choice(unconnected_colors)
              start, end = endpoints[selected_color]
              start_x, start_y = start
              end_x, end_y = end

              # 用 BFS 生成一條路徑
              from collections import deque

              queue = deque([(start_x, start_y, [])])  # (當前位置, 路徑)
              visited = set()

              path = None
              while queue:
                  current_x, current_y, current_path = queue.popleft()

                  # 如果到達終點，記錄路徑
                  if (current_x, current_y) == (end_x, end_y):
                      path = current_path
                      break

                  if (current_x, current_y) in visited:
                      continue
                  visited.add((current_x, current_y))

                  # 搜索相鄰格子
                  for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                      nx, ny = current_x + dx, current_y + dy
                      if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                          if individual.genotype[nx, ny] in ("", selected_color):  # 空格或相同顏色
                              queue.append((nx, ny, current_path + [(nx, ny)]))

              # 更新網格
              if path:
                  for x, y in path:
                      individual.genotype[x, y] = selected_color
        # ensure requriment colors
        for color, length in color_path_lengths.items():
            if color != selected_color:
                empty_positions = np.where(individual.genotype == "")
                i = 0
                for pos in zip(empty_positions[0], empty_positions[1]):
                    if i  == length+1:
                        break
                    individual.genotype[pos] = random.choice(COLORS)
                    i += 1

        # visualize_solution(individual.genotype)

                    
        # 將剩下的格子隨機填充，以增加多樣性
        empty_positions = np.where(individual.genotype == "")
        for pos in zip(empty_positions[0], empty_positions[1]):
            individual.genotype[pos] = random.choice(COLORS)
        
        enforce_fixed_points(individual)
        population.append(individual)
        # visualize_solution(individual.genotype)

    return population

# 父母選擇：競賽選擇
def tournament_selection(population):
    selected = random.sample(population, TOURNAMENT_SIZE)
    return max(selected, key=lambda ind: ind.fitness)

# 交叉：單點交叉
def find_fixed_points(endpoints,fix):
    """
    找出所有顏色的起點和終點，返回一個保護點的集合。
    """
    fixed = set()
    for color, (start, end) in endpoints.items():
        fixed.add(start)
        fixed.add(end)
    for color, points in fix.items():
        for point in points:
            fixed.add(point)
    return fixed

def find_initial_fix(individual):
    global fix
    # 把indfivial row col 左右各增加一行
    rows, cols = individual.genotype.shape
    # 把矩陣都填入1
    grid_test = np.full((rows + 2, cols + 2), "1", dtype=str)
    grid_test[1:-1, 1:-1] = individual.genotype
    # 便利這個grid_test如果上下左右其中有三個不是空白就檢查是不是附近有唯一一個顏色，如果有就填入那個顏色
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if grid_test[i, j] != "":  # 如果当前格子不是空白
                # 获取上下左右的值
                left, right, down, up = grid_test[i-1, j], grid_test[i+1, j], grid_test[i, j-1], grid_test[i, j+1]
                counts = [left, right, down, up].count("")
                
                if counts == 1:  # 如果周围只有一个空格
                    if left == "":
                        grid_test[i-1, j] = grid_test[i, j]
                        new_point = (i-2, j-1)
                    elif right == "":
                        grid_test[i+1, j] = grid_test[i, j]
                        new_point = (i, j-1)
                    elif down == "":
                        grid_test[i, j-1] = grid_test[i, j]
                        new_point = (i-1, j-2)
                    elif up == "":
                        grid_test[i, j+1] = grid_test[i, j]
                        new_point = (i-1, j)
                    
                    # 更新 fix
                    color = grid_test[i, j]
                    if color not in fix:
                        fix[color] = set()  # 使用集合避免重复
                    fix[color].add(new_point)

    final_fix = copy.deepcopy(fix)
    # 換serch fix 附近有沒有可以填入的顏色
    while True:
        update = False
        new_fix = {color: set() for color in fix}  # 初始化新的 fix

        for color, points in fix.items():
            for x, y in points:
                # 转换到 grid_test 坐标
                grid_x, grid_y = x + 1, y + 1
                
                # 获取邻居的值
                neighbors = {
                    "left": (grid_x - 1, grid_y),
                    "right": (grid_x + 1, grid_y),
                    "down": (grid_x, grid_y - 1),
                    "up": (grid_x, grid_y + 1),
                }
                
                # 找到空白的邻居
                empty_neighbors = {k: v for k, v in neighbors.items() if grid_test[v] == ""}
                
                if len(empty_neighbors) == 1:  # 如果只有一个空白邻居
                    update = True
                    direction, (nx, ny) = empty_neighbors.popitem()
                    grid_test[nx, ny] = color  # 填入颜色
                    
                    # 更新新固定点
                    new_point = (nx - 1, ny - 1)  # 转回 fix 坐标
                    new_fix[color].add(new_point)
                    final_fix[color].add(new_point)
        
        if not update:  # 如果没有任何更新，跳出循环
            break
        fix = new_fix

    fix = final_fix
    # 打印最终状态
    # 將填充後的矩陣放回individual
    individual.genotype = grid_test[1:-1, 1:-1]               
    

def crossover(parent1, parent2):
    """
    二維空間的交叉操作，保護起點和終點位置。
    """
    fixed_points = find_fixed_points(endpoints,fix)
    child1_genotype = copy.deepcopy(parent1.genotype)
    child2_genotype = copy.deepcopy(parent2.genotype)

    # 隨機選擇水平或垂直切割
    if random.random() < 0.5:
        # 水平切割 (交換部分行)
        row = random.randint(0, GRID_SIZE - 1)
        for col in range(GRID_SIZE):
            if (row, col) not in fixed_points:
                child1_genotype[row, col], child2_genotype[row, col] = (
                    child2_genotype[row, col],
                    child1_genotype[row, col],
                )
    else:
        # 垂直切割 (交換部分列)
        col = random.randint(0, GRID_SIZE - 1)
        for row in range(GRID_SIZE):
            if (row, col) not in fixed_points:
                child1_genotype[row, col], child2_genotype[row, col] = (
                    child2_genotype[row, col],
                    child1_genotype[row, col],
                )

    # 建立新的子代個體
    child1 = Individual()
    child2 = Individual()
    child1.genotype = child1_genotype
    child2.genotype = child2_genotype

    return child1, child2

# 突變：隨機區段重組
# 重新設計突變函數，以確保不會破壞起點和終點
def mutate(individual):
    if random.random() < MUTATION_RATE:
        # 找出所有固定的起點和終點位置
        fixed_points = find_fixed_points(endpoints,fix)

        # 隨機選擇一個區域
        x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        size = random.randint(1, GRID_SIZE//2)  # 隨機區段大小
        end_x, end_y = min(x + size, GRID_SIZE), min(y + size, GRID_SIZE)

        # 收集非固定點的顏色位置
        mutable_positions = []
        mutable_colors = []

        for i in range(x, end_x):
            for j in range(y, end_y):
                if (i, j) not in fixed_points:
                    mutable_positions.append((i, j))
                    mutable_colors.append(individual.genotype[i, j])

        # 隨機打亂非固定點的顏色
        if mutable_colors:
            random.shuffle(mutable_colors)

            # 將打亂後的顏色放回非固定點位置
            for (i, j), color in zip(mutable_positions, mutable_colors):
                individual.genotype[i, j] = color
# def visualize_solution_dynamic(genotype, fig=None, ax=None):
#     # 如果没有提供图形和轴，则初始化
#     if fig is None or ax is None:
#         plt.ion()  # 开启交互模式
#         fig, ax = plt.subplots(figsize=(6, 6))
#         ax.set_xlim(0, GRID_SIZE)
#         ax.set_ylim(0, GRID_SIZE)
#         ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1))
#         ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1))
#         ax.grid(True)
#         ax.invert_yaxis()  # 反转 y 轴，确保 (0, 0) 在左上角

#     # 移除已有的补丁
#     for patch in ax.patches:
#         patch.remove()
#     # 移除已有的文本
#     for text in ax.texts:
#         text.remove()

#     # 遍历每个格子，填充颜色
#     for i in range(GRID_SIZE):
#         for j in range(GRID_SIZE):
#             color = genotype[i, j]
#             rect = plt.Rectangle((j, i), 1, 1, facecolor=COLOR_MAP[color], edgecolor='black')
#             ax.add_patch(rect)

#             # 标记起点和终点
#             for c, (start, end) in endpoints.items():
#                 if (i, j) == start or (i, j) == end:
#                     ax.text(j + 0.5, i + 0.5, c, color='black', ha='center', va='center', fontsize=12, fontweight='bold')

#     # 刷新画布
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     # 暫停一段時間，以便觀察
#     # plt.pause(1)
#     return fig, ax  # 返回图形和轴对象，供后续更新
def visualize_solution_dynamic(genotype, fig=None, ax=None):
    """
    动态更新网格解的图表
    """
    # 移除已有的补丁和文本（清空旧内容）
    for patch in ax.patches:
        patch.remove()
    for text in ax.texts:
        text.remove()

    # 遍历每个格子，填充颜色
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color = genotype[i, j]
            rect = plt.Rectangle((j, i), 1, 1, facecolor=COLOR_MAP[color], edgecolor='black')
            ax.add_patch(rect)

            # 标记起点和终点
            for c, (start, end) in endpoints.items():
                if (i, j) == start or (i, j) == end:
                    ax.text(j + 0.5, i + 0.5, c, color='black', ha='center', va='center', fontsize=12, fontweight='bold')

    # 设置网格和坐标轴范围
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1))
    ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1))
    ax.grid(True)
    ax.invert_yaxis()  # 反转 y 轴，确保 (0, 0) 在左上角

    # 刷新画布
    fig.canvas.draw()
    fig.canvas.flush_events()

# 演化循環
def evolve(population):
    # 初始化绘图窗口
    fig, axs = plt.subplots(2, 1, figsize=(4, 7))
    fig.subplots_adjust(hspace=0.5)  # 调整两个子图的间距

    fitness_history = []  # 用于记录每一代的最佳适应度

    for generation in range(GENERATIONS):
        # 计算适应度
        for individual in population:
            individual.calculate_fitness()
            
            # if individual.finished:
            #     print(f"Solution found in generation {generation + 1}!")
            #     fig, ax = visualize_solution_dynamic(individual.genotype, fig, ax)
            #     plt.ioff()  # 关闭交互模式
            #     plt.show()  # 显示最终结果
            #     return  # 提前结束演化函数

        # 生成新族群
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = crossover(parent1, parent2)  # 正确接收两个子代
            mutate(child1)
            mutate(child2)
            # 检查 child fitness = 0 的直接不加入
            if child1.fitness != 0:
                new_population.append(child1)
            if child2.fitness != 0:
                new_population.append(child2)

        # 更新族群
        population = new_population
        for individual in population:
            individual.calculate_fitness()

        # 输出每代的最佳适应度
        best_fitness = max(individual.fitness for individual in population)
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")
        fitness_history.append(best_fitness)
        best_individual = max(population, key=lambda ind: ind.fitness)
        # 如果找到最优解，提前结束
        if best_individual.finished:
            print(f"Solution found in generation {generation + 1}!")
            break
        # 动态更新画面
        # fig, ax = visualize_solution_dynamic(best_individual.genotype, fig, ax)
        visualize_solution_dynamic(best_individual.genotype, fig, axs[0])

        # 动态更新适应度曲线
        axs[1].clear()
        axs[1].plot(range(len(fitness_history)), fitness_history, marker='o', label="Best Fitness")
        axs[1].set_title("Fitness Over Generations")
        axs[1].set_xlabel("Generation")
        axs[1].set_ylabel("Fitness")
        axs[1].legend()
        axs[1].grid(True)
        plt.pause(0.01)  # 暂停一小段时间，以便观察

    # 输出最终解
    print("Best Solution:")
    print(best_individual.genotype)

    # 显示最终适应度曲线
    axs[1].plot(range(len(fitness_history)), fitness_history, marker='o', label="Best Fitness")
    axs[1].set_title("Fitness Over Generations")
    axs[1].set_xlabel("Generation")
    axs[1].set_ylabel("Fitness")
    axs[1].legend()
    axs[1].grid(True)

    plt.ioff()  # 关闭交互模式
    plt.show()  # 显示最终结果
# 主程式
population = create_population(POPULATION_SIZE)
evolve(population)