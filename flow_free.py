import numpy as np
import random
from collections import deque
import copy
import matplotlib.pyplot as plt

# 定義網格大小（例如5x5）
GRID_SIZE = 7
NUM_COLORS = 6  # 假設我們有五對顏色點

# 初始化參數
POPULATION_SIZE = 500
GENERATIONS = 1000
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 5

# 初始設定的顏色
COLORS = ["R", "G", "B", "Y", "O","P"]  # 紅、綠、藍、黃、橙
color_path_lengths = {}  # 每個顏色的最短路徑長度
COLOR_MAP = {
    "R": "red",
    "G": "green",
    "B": "blue",
    "Y": "yellow",
    "O": "orange",
    "P": "purple",
    "": "white"  # 空白格子
}

# 隨機生成起點和終點
endpoints = {
    "R": [(1, 6), (5, 4)],  # Yellow
    "B": [(0, 6), (6, 5)],  # Blue
    "G": [(3, 3), (4, 2)],  # Green
    "Y": [(4, 4), (5, 5)],  # Red
    "O": [(1, 5), (2, 1)],  # Orange
    "P": [(3, 4), (6, 6)]   # purple
}
fix = {}
# 定義個體
class Individual:
    def __init__(self):
        # self.genotype = np.random.choice(COLORS, (GRID_SIZE, GRID_SIZE))
        # 生成一個空白網格
        self.genotype = np.full((GRID_SIZE, GRID_SIZE), "", dtype=str)
        self.fitness = -1
        self.final = 5
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
        for color, (start, end) in endpoints.items():
            if self.is_connected(color, start, end):
                path_score += 10
                color_connected.append(color)
            else:
                connection_penalty += 5  # 每個未連線的顏色降低 5 分
        
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

def find_initial_fix(individual):
            
    # 把indfivial row col 左右各增加一行
    rows, cols = individual.genotype.shape
    # 把矩陣都填入1
    grid_test = np.full((rows + 2, cols + 2), "1", dtype=str)
    grid_test[1:-1, 1:-1] = individual.genotype
    # 便利這個grid_test如果上下左右其中有三個不是空白就檢查是不是附近有唯一一個顏色，如果有就填入那個顏色
    for i in range(1, rows+1):
        for j in range(1, cols+1):
            if grid_test[i, j] != "":
                left,right,down,up = grid_test[i-1, j], grid_test[i+1, j], grid_test[i, j-1], grid_test[i, j+1]
                counts = [left, right, down, up].count("")
                if counts == 1:
                    if left == "":
                        grid_test[i-1, j] = grid_test[i, j]
                        ## 更新fix
                        if grid_test[i-1, j] not in fix:
                            fix[grid_test[i-1, j]] = []
                            fix[grid_test[i-1, j]].append((i-2, j-1))
                    elif right == "":
                        grid_test[i+1, j] = grid_test[i, j]
                        ## 更新fix
                        if grid_test[i+1, j] not in fix:
                            fix[grid_test[i+1, j]] = []
                            fix[grid_test[i+1, j]].append((i, j-1))
                    elif down == "":
                        grid_test[i, j-1] = grid_test[i, j]
                        ## 更新fix
                        if grid_test[i, j-1] not in fix:
                            fix[grid_test[i, j-1]] = []
                            fix[grid_test[i, j-1]].append((i-1, j-2))
                    elif up == "":
                        grid_test[i, j+1] = grid_test[i, j]
                        ## 更新fix
                        if grid_test[i, j+1] not in fix:
                            fix[grid_test[i, j+1]] = []
                            fix[grid_test[i, j+1]].append((i-1, j))

    # 換serch fix 附近有沒有可以填入的顏色
    
    while True:
        update = False
        for color, points in fix.items():
            for point in points:
                x, y = point
                x = x + 1
                y = y + 1
                left,right,down,up = grid_test[x-1, y], grid_test[x+1, y], grid_test[x, y-1], grid_test[x, y+1]
                counts = [left, right, down, up].count("")
                if counts == 1:
                    update = True
                    if left == "":
                        grid_test[x-1, y] = color
                        ## 更新fix
                        if grid_test[x-1, y] not in fix:
                            fix[grid_test[x-1, y]] = []
                            fix[grid_test[x-1, y]].append((x-2, y-1))
                    elif right == "":
                        grid_test[x+1, y] = color
                        ## 更新fix
                        if grid_test[x+1, y] not in fix:
                            fix[grid_test[x+1, y]] = []
                            fix[grid_test[x+1, y]].append((x, y-1))
                    elif down == "":
                        grid_test[x, y-1] = color
                        ## 更新fix
                        if grid_test[x, y-1] not in fix:
                            fix[grid_test[x, y-1]] = []
                            fix[grid_test[x, y-1]].append((x-1, y-2))
                    elif up == "":
                        grid_test[x, y+1] = color
                        ## 更新fix
                        if grid_test[x, y+1] not in fix:
                            fix[grid_test[x, y+1]] = []
                            fix[grid_test[x, y+1]].append((x-1, y))
        if update == False:
            break
            
    # 將填充後的矩陣放回individual
    individual.genotype = grid_test[1:-1, 1:-1]               
                    
            

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

        # 計算有沒有可以固定的顏色點
        find_initial_fix(individual)
        # visualize_solution(individual.genotype)
        # exit()

        # 分配顏色到網格上，確保每種顏色至少有最短路徑長度的格子數
        for color, length in color_path_lengths.items():
            positions = random.sample(range(GRID_SIZE * GRID_SIZE), length)  # 隨機選擇網格位置
            for pos in positions:
                x, y = divmod(pos, GRID_SIZE)
                individual.genotype[x, y] = color
        
        # 將剩下的格子隨機填充，以增加多樣性
        empty_positions = np.where(individual.genotype == "")
        for pos in zip(empty_positions[0], empty_positions[1]):
            individual.genotype[pos] = random.choice(COLORS)
        
        enforce_fixed_points(individual)
        population.append(individual)

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

def visualize_solution(genotype):
    plt.figure(figsize=(6, 6))
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
    plt.show()

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
        size = random.randint(1, 3)  # 隨機區段大小
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

# 演化循環
def evolve(population):
    for generation in range(GENERATIONS):
        # 計算適應度
        for individual in population:
            individual.calculate_fitness()
            
            if individual.finished:
                print(f"Solution found in generation {generation + 1}!")
                visualize_solution(individual.genotype)
                return  # 提前結束演化函數

        # 生成新族群
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = crossover(parent1, parent2)  # 正確接收兩個子代
            mutate(child1)
            mutate(child2)
            # 這邊檢查child fitness = 0 的直接不加入
            if child1.fitness != 0:
                new_population.append(child1)
            if child2.fitness != 0:
                new_population.append(child2)

        # 更新族群
        population = new_population
        for individual in population:
            individual.calculate_fitness()

        # 輸出每代的最佳適應度
        best_fitness = max(individual.fitness for individual in population)
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    # 輸出最終解
    best_individual = max(population, key=lambda ind: ind.fitness)
    print("Best Solution:")
    print(best_individual.genotype)

    #可視化最終解
    visualize_solution(best_individual.genotype)

# 主程式
population = create_population(POPULATION_SIZE)
evolve(population)