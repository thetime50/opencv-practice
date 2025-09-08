import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import math

SATASET_FILE = os.path.join(os.path.dirname(__file__), 'output')
MODEL_FILE = os.path.join(SATASET_FILE, 'sliding_00.h5')

import numpy as np
import random
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

class SlidingPuzzleEnv:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.size = self.m * self.n
        self.empty_tile = self.size - 1
        self.reset()

    def reset(self,scramble_steps=0):
        self.state = list(range(self.size))
        action = None
        for _ in range(scramble_steps):
            actions = self.get_moves( [self.negative_action(action)] if action else [])
            action = random.choice(actions)
            self.do_action(action)
        return self._get_obs()

    def _get_obs(self):
        # mat = np.array(self.state).reshape((self.m, self.n)) / (self.size - 1)
        # return mat[..., np.newaxis].astype(np.float32)  # shape: (m,n,1)
        one_hot = np.eye(self.size)[np.array(self.state,dtype=np.uint8)]
        return one_hot.reshape( self.m, self.n, self.size)

    def get_moves(self,excludes = None):
        idx = self.state.index(self.empty_tile)
        row, col = divmod(idx, self.n)
        moves = []
        if row > 0 and (excludes is None or 0 not in excludes):
            moves.append(0)  # 上
        if row < self.m - 1 and (excludes is None or 1 not in excludes):
            moves.append(1)  # 下
        if col > 0 and (excludes is None or 2 not in excludes):
            moves.append(2)  # 左
        if col < self.n - 1 and (excludes is None or 3 not in excludes):
            moves.append(3)  # 右
        return moves

    def _calculate_total_distance(self,state):
        """计算所有滑块到正确位置的曼哈顿距离之和"""
        total_distance = 0
        for i, tile in enumerate(state):
            if tile == self.empty_tile:  # 空滑块不计算距离
                continue
            # 计算当前滑块应该在的正确位置
            correct_row, correct_col = divmod(tile, self.n)
            current_row, current_col = divmod(i, self.n)
            distance = abs(current_row - correct_row) + abs(current_col - correct_col)
            total_distance += distance
        return total_distance
    def adjacent_distance_max(self):
        m=self.m
        n=self.n
        return (m*(n-1)+n*(m-1))*((m-1) + (n-1) - 1)
    def calculate_adjacent_distance(self, state):
        """
        计算所有原本相邻方块（横向 & 纵向）的曼哈顿间距总和
        忽略空格
        """
        total_distance = 0 
        # 数字 → 当前位置索引
        pos = [0] * self.size
        for idx, tile in enumerate(state):
            pos[tile] = idx

        # 横向相邻对
        for row in range(self.m):
            for col in range(self.n - 1):
                a = row * self.n + col
                b = a + 1
                if a != self.size - 1 and b != self.size - 1:
                    ax, ay = divmod(pos[a], self.n)
                    bx, by = divmod(pos[b], self.n)
                    total_distance += abs(ax - bx) + abs(ay - by) -1

        # 纵向相邻对
        for row in range(self.m - 1):
            for col in range(self.n):
                a = row * self.n + col
                b = a + self.n
                if a != self.size - 1 and b != self.size - 1:
                    ax, ay = divmod(pos[a], self.n)
                    bx, by = divmod(pos[b], self.n)
                    total_distance += abs(ax - bx) + abs(ay - by) -1

        return total_distance

    def negative_action(self,action):
        m = {
            0:1,
            1:0,
            2:3,
            3:2,
        }
        return m.get(action)

    def do_action(self,action):
        idx = self.state.index(self.empty_tile)
        if action == 0: target = idx - self.n
        elif action == 1: target = idx + self.n
        elif action == 2: target = idx - 1
        elif action == 3: target = idx + 1

        self.state[idx], self.state[target] = self.state[target], self.state[idx]

    def step(self, action):
        # 这里的评分加上gamma<1 让距离越远的分数越小，不要在远的地方刷分数
        legal_moves = self.get_moves()
        reward = -0.01
        done = False
        if action not in legal_moves:
            reward = -1
            # reward = - math.square(self.m+self.n)
        else:
            
            adjacent_max = self.adjacent_distance_max()
            before_total_distance = self._calculate_total_distance(self.state)
            # before_adjacent_distance = self.calculate_adjacent_distance(self.state)/adjacent_max
            self.do_action(action)

            # 计算所有滑块到正确位置的曼哈顿距离之和
            total_distance = self._calculate_total_distance(self.state)
            # adjacent_distance = self.calculate_adjacent_distance(self.state)/adjacent_max
            # 使用距离之和作为额外奖励（负奖励，因为距离越小越好）
            if(total_distance > before_total_distance):
                reward = 0.06 # * before_total_distance-total_distance
            else:
                reward = -0.03 # * before_total_distance-total_distance
            # distance_reward = -0.01 * total_distance  # 缩放因子可根据需要调整
            # reward = distance_reward
            # if(adjacent_distance > before_adjacent_distance):
            #     reward = 0.02 # * before_total_distance-total_distance
            # else:
            #     reward = -0.01 # * before_total_distance-total_distance

            if self.state == list(range(self.size)):
                reward = 1.0
                # reward = math.pow(self.m+self.n,4)
                done = True
        return self._get_obs(), reward, done

# ===== DQN Agent =====
class DQNAgent:
    def __init__(self, action_size,m,n):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9 # 越外面分值越小，禁止在外面刷分
        # self.epsilon = 1.0
        # self.epsilon_decay = 0.995
        self.epsilon = 0.2
        self.epsilon_decay = 1
        self.epsilon_min = 0.1
        self.m = m
        self.n = n
        self.size = m*n
        
        # self.optimizer = tf.keras.optimizers.Adam(1e-3)
        # 动态学习率 指数衰减
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=8e-5, decay_steps=1, decay_rate=0.995)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # 或者使用余弦退火
        # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        #     initial_learning_rate=1e-5, decay_steps=1000
        # )
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.model = self._build_model()

    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.m, self.n, self.size))

        x = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(8192, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.action_size, activation='linear')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=self.optimizer, loss='mse')
        return model

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def act(self, state, legal_moves):
        if np.random.rand() < self.epsilon:
            return random.choice(legal_moves)
        q = self.model.predict(state[np.newaxis,...], verbose=0)[0]
        # q_masked = [q[a] for a in legal_moves]
        # return legal_moves[np.argmax(q_masked)]
        return np.argmax(q)

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size: return
        batch = random.sample(self.memory, batch_size)
        states, targets = [], []
        for s, a, r, s2, done in batch:
            target = r
            if not done:
                q_next = self.model.predict(s2[np.newaxis,...], verbose=0)[0]
                target += self.gamma * np.max(q_next)
            q_current = self.model.predict(s[np.newaxis,...], verbose=0)[0]
            q_current[a] = target
            states.append(s)
            targets.append(q_current)
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

    def test_memory(self, cnt):
        test_list = random.sample(self.memory,cnt) if len(self.memory)>cnt else self.memory.copy()
        success_cnt=0
        for i,item in enumerate(test_list):
            a = self.model.predict(item[0][np.newaxis],verbose=0)[0]
            if(np.argmax(a) == item[1]and item[2]>0 or np.argmax(a) != item[1]and item[2]<0 ): success_cnt += 1
        return success_cnt / cnt


class SlidingPuzzleDetector:
    def __init__(self, m=3, n=3,model=None):
        self.env = SlidingPuzzleEnv(m, n)
        self.model = model if model else tf.keras.models.load_model(MODEL_FILE)
        self.action_names = ['上', '下', '左', '右']

    def state2str(self,state):
        return ','.join([str(i) for i in state])
    
    def solve(self, initial_state, max_steps=50):
        """
        解决数码难题
        :param initial_state: 初始状态数组
        :param max_steps: 最大尝试步数
        :return: (success, path) - 成功标志和解决路径
        """
        # 设置环境状态
        self.env.state = initial_state.copy()
        path = []
        visited_states = set()
        visited_states.add(self.state2str(self.env.state))  # 添加初始状态
        
        # 定义相反动作映射
        opposite_actions = {0: 1, 1: 0, 2: 3, 3: 2}  # 上↔下, 左↔右
        
        for step in range(max_steps):
            # 检查是否已经解决
            if self.env.state == list(range(self.env.size)):
                return True, path
            
            # 获取当前状态观察值
            state_obs = self.env._get_obs()
            legal_moves = self.env.get_moves()
            
            # 使用模型预测所有动作的Q值
            q_values = self.model.predict(state_obs[np.newaxis, ...], verbose=0)[0]
            
            # 按Q值从高到低排序所有动作
            all_actions = list(range(4))
            sorted_actions = sorted(all_actions, key=lambda a: q_values[a], reverse=True)
            
            # 过滤掉上一次动作的相反动作（如果存在上一次动作）
            if path:
                last_action = path[-1]
                opposite_action = opposite_actions.get(last_action)
                # 从合法动作中移除相反动作
                if opposite_action in legal_moves:
                    legal_moves.remove(opposite_action)
            
            # 选择不会导致循环状态的最高Q值合法动作
            chosen_action = None
            next_state_tuple = None
            
            for action in sorted_actions:
                if action not in legal_moves:
                    continue
                    
                # 模拟执行动作来检查下一个状态
                temp_env = SlidingPuzzleEnv(self.env.m, self.env.n)
                temp_env.state = self.env.state.copy()
                next_state, reward, done = temp_env.step(action)
                next_state_tuple = temp_env.state
                
                # 检查下一个状态是否已经访问过
                if self.state2str(next_state_tuple) not in visited_states:
                    chosen_action = action
                    break
            
            if chosen_action is None:
                # 所有可能的动作都会导致循环状态
                return False, path
            
            # 执行选择的动作
            next_state, reward, done = self.env.step(chosen_action)
            next_state_tuple = tuple(self.env.state)
            visited_states.add(self.state2str(next_state_tuple))
            path.append(chosen_action)
            
            if done:
                return True, path
        
        return False, path
    
    def get_state_string(self, state):
        """将状态数组转换为可读字符串"""
        grid = np.array(state).reshape(self.env.m, self.env.n)
        return '\n'.join([' '.join('  ' if x == self.env.empty_tile else f'{x:2d}' for x in row) for row in grid])
    
    def apply_action_sequence(self, state, actions):
        """应用动作序列到状态"""
        current_state = state.copy()
        env_copy = SlidingPuzzleEnv(self.env.m, self.env.n)
        env_copy.state = current_state
        
        for action in actions:
            if action not in env_copy.get_moves():
                return None
            env_copy.step(action)
        
        return env_copy.state
    
    def test_performance(self,cnt,scrambled_step):
        # 重置环境并随机打乱n步
        env = SlidingPuzzleEnv(self.env.m,self.env.n)
        success_cnt = 0
        for test_idx in range(cnt):
            env.reset(scrambled_step)
            scrambled_state = env.state.copy()
            
            # 使用solve求解
            success, solution_path = self.solve(scrambled_state, max_steps=scrambled_step*3)
            if success: success_cnt+=1
        return success_cnt/cnt

if __name__ == '__main__':

    # ===== 训练 =====
    episodes = 500
    reward_list = []
    puzzle_size = {'m':3,'n':3}

    # 初始化绘图
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlim(0, episodes)
    # ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DQN Training Reward")

    agent = DQNAgent(action_size=4,**puzzle_size)

    env = SlidingPuzzleEnv(**puzzle_size)
    # def get_fn(x1,y1,x2,y2):
    #     k = (y2-y1)/(x2-x1)
    #     b = y1-k*x1
    #     return lambda x:k*x+b
    # kxb = get_fn(0,2,150,50)
    # for e in range(episodes):
    #     scramble_steps = min(math.floor( kxb(e)),50)

    e = 0
    total_cnt = 0
    while e < episodes:
        total_cnt += 1
        scramble_steps = max(2,min(e,70))
        state = env.reset(scramble_steps)
        total_reward = 0
        action = None
        solve_step = min(50,scramble_steps*5)
        for t in range(solve_step):
            legal = env.get_moves()
            action = agent.act(state, legal)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done: break
        # for i in range(10):
        agent.replay(batch_size=64)
        reward_list.append(total_reward)

        # ===== 实时更新绘图 =====
        line.set_data(range(len(reward_list)), reward_list)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.05)

        if total_cnt % 5 == 0:
            agent.model.save(MODEL_FILE)
            detector = SlidingPuzzleDetector(model = tf.keras.models.clone_model(agent.model), **puzzle_size)
            success_rate = detector.test_performance(20,scramble_steps)
            mem_rate = agent.test_memory(50)
            print(f"Episode {total_cnt} {e+1}/{episodes}, scramble: {scramble_steps}, mem_rate: {mem_rate*100:.1f}%, success: {success_rate*100:.1f}%, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            if(success_rate>=0.9):
                e+=1

    agent.model.save(MODEL_FILE)

    # ===== 绘制奖励曲线 =====
    plt.plot(reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Reward")
    plt.show()
