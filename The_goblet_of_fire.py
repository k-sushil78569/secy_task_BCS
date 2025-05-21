import pygame
import numpy as np
from collections import defaultdict
from queue import Queue
ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_MIN = 0.1
DECAY_RATE = 0.995
EPISODES = 5000
GRID_SIZE = 40
MAZE = [
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 2, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
]

class MazeEnvironment:
    def __init__(self):
        self.grid = MAZE
        self.cup_pos = (2, 5)  #now directly given but can do through the matrix and find it
        self.harry_pos = None
        self.death_eater_pos = None
        self._place_entities()
        
    def _place_entities(self):
        empty_cells = [(i,j) for i in range(len(self.grid)) 
                      for j in range(len(self.grid[0])) 
                      if self.grid[i][j] == 0 and (i,j) != self.cup_pos]
        np.random.shuffle(empty_cells)
        self.harry_pos = empty_cells[0]
        self.death_eater_pos = empty_cells[1]
    
    def get_state(self):
        return (*self.harry_pos, *self.death_eater_pos, *self.cup_pos)
    
    def move_entity(self, entity, action):
        x, y = entity
        dx, dy = 0, 0
        if action == 0: dx = -1  # Up
        elif action == 1: dx = 1  # Down
        elif action == 2: dy = -1  # Left
        elif action == 3: dy = 1  # Right
        
        new_x = x + dx
        new_y = y + dy
        if self._is_valid_move(new_x, new_y):
            return (new_x, new_y)
        return (x, y)
    
    def _is_valid_move(self, x, y):
        return 0 <= x < len(self.grid) and 0 <= y < len(self.grid[0]) and self.grid[x][y] != 1

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.epsilon = EPSILON_START
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        new_q = (1 - ALPHA) * current_q + ALPHA * (reward + GAMMA * max_future_q)
        self.q_table[state][action] = new_q
        
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * DECAY_RATE)

def bfs_path(start, target, grid):
    q = Queue()
    q.put(start)
    visited = {start: None}
    
    while not q.empty():
        current = q.get()
        if current == target:
            break
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            next_pos = (current[0] + dx, current[1] + dy)
            if 0 <= next_pos[0] < len(grid) and 0 <= next_pos[1] < len(grid[0]):
                if grid[next_pos[0]][next_pos[1]] != 1 and next_pos not in visited:
                    q.put(next_pos)
                    visited[next_pos] = current
    path = []
    current = target
    while current != start:
        path.append(current)
        current = visited[current]
    return path[-1] if path else start

def train():
    pygame.init()
    env = MazeEnvironment()
    agent = QLearningAgent(env)
    success_count = 0
    
    screen = pygame.display.set_mode((len(env.grid[0])*GRID_SIZE, len(env.grid)*GRID_SIZE))
    pygame.display.set_caption("Triwizard Maze Challenge")
    font = pygame.font.SysFont(None, 24)

    for episode in range(EPISODES):
        state = env.get_state()
        done = False
        total_reward = 0
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.choose_action(state)
            env.harry_pos = env.move_entity(env.harry_pos, action)

            next_de_pos = bfs_path(env.death_eater_pos, env.harry_pos, env.grid)
            env.death_eater_pos = next_de_pos
            
            if env.harry_pos == env.cup_pos:
                reward = 10
                success_count += 1
                done = True
            elif env.harry_pos == env.death_eater_pos:
                reward = -10
                success_count = 0
                done = True
            else:
                reward = -0.1
                if env.harry_pos == state[:2]:
                    reward -= 1  
            
            
            next_state = env.get_state()
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
            
            screen.fill((255,255,255))
            for i in range(len(env.grid)):
                for j in range(len(env.grid[0])):
                    color = (0,0,0) if env.grid[i][j] == 1 else (255,255,255)
                    if (i,j) == env.cup_pos: color = (0,255,0)
                    pygame.draw.rect(screen, color, (j*GRID_SIZE, i*GRID_SIZE, GRID_SIZE, GRID_SIZE), 0)
            
            
            pygame.draw.circle(screen, (255,0,0), 
                             (env.harry_pos[1]*GRID_SIZE + GRID_SIZE//2, 
                              env.harry_pos[0]*GRID_SIZE + GRID_SIZE//2), 15)
            pygame.draw.rect(screen, (0,0,255), 
                            (env.death_eater_pos[1]*GRID_SIZE, 
                             env.death_eater_pos[0]*GRID_SIZE, GRID_SIZE, GRID_SIZE))
            
            
            text = font.render(f"Episode: {episode} | Reward: {total_reward:.1f}", True, (0,0,0))
            screen.blit(text, (10, 10))
            pygame.display.flip()
        
        agent.decay_epsilon()
        if success_count >= 10:
            print(f"Training complete! Harry escaped 10 times consecutively at episode {episode}")
            break
    
    
    np.save("trained_q_table.npy", dict(agent.q_table))
    pygame.quit()

if __name__ == "__main__":
    train()
