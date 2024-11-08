import numpy as np
import random
from collections import deque

# 定義網格大小（例如5x5）
GRID_SIZE = 5
NUM_COLORS = 5  # 假設我們有五對顏色點

# 初始化參數
POPULATION_SIZE = 50
GENERATIONS = 1000
MUTATION_RATE = 0.01
TOURNAMENT_SIZE = 5

# 初始設定的顏色
COLORS = ["R", "G", "B", "Y", "O"]  # 紅、綠、藍、黃、橙

# 隨機生成起點和終點
endpoints = {
    "Y": [(0, 1), (3, 0)],  # Yellow
    "B": [(0, 2), (4, 0)],  # Blue
    "G": [(0, 3), (4, 3)],  # Green
    "R": [(1, 3), (2, 2)],  # Red
    "O": [(3, 3), (4, 2)],  # Orange
}
# 定義個體
class Individual:
    def __init__(self):
        self.genotype = np.random.choice(COLORS, (GRID_SIZE, GRID_SIZE))
        self.fitness = 0
        self.final = 5
        self.finished = False

    def calculate_fitness(self):
        cluster_num = self.compute_cluster(self.genotype)
        
        if cluster_num > self.final:
            self.fitness += (1/(cluster_num - self.final))* 10
        elif cluster_num == self.final:
            self.finished = True
            return
        
        # 檢查每個顏色的起點與終點是否連通
        path_score = 0
        for color, (start, end) in endpoints.items():
            if self.is_connected(color, start, end):
                path_score += 1
        
        # 假設適應度為覆蓋得分和路徑連通性得分的和
        self.fitness += path_score


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

# 初始族群生成
def create_population(size):
    # 計算每個顏色至少需要的格子數（最短路徑長度）
    color_path_lengths = {}
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
        
        population.append(individual)

    return population

# 父母選擇：競賽選擇
def tournament_selection(population):
    selected = random.sample(population, TOURNAMENT_SIZE)
    return max(selected, key=lambda ind: ind.fitness)

# 交叉：單點交叉
# 找到能交叉但不會破壞起點和終點的位置
def crossover(parent1, parent2):
    return NotImplementedError

# 突變：隨機區段重組
# 重新設計突變函數，以確保不會破壞起點和終點
def mutate(individual):
    if random.random() < MUTATION_RATE:
        # 隨機選擇一個區域並打亂其顏色
        x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
        size = random.randint(1, 3)  # 隨機區段大小
        end_x, end_y = min(x + size, GRID_SIZE), min(y + size, GRID_SIZE)
        subgrid = individual.genotype[x:end_x, y:end_y]
        np.random.shuffle(subgrid.flat)  # 隨機打亂選定區域
        individual.genotype[x:end_x, y:end_y] = subgrid

# 演化循環
def evolve(population):
    for generation in range(GENERATIONS):
        # 計算適應度
        for individual in population:
            individual.calculate_fitness()

        # 生成新族群
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = crossover(parent1, parent2)
            mutate(child)
            new_population.append(child)

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

# 主程式
population = create_population(POPULATION_SIZE)
evolve(population)
