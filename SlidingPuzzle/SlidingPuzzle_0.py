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

    def reset(self):
        self.state = list(range(self.size))
        random.shuffle(self.state)
        return self._get_obs()

    def _get_obs(self):
        mat = np.array(self.state).reshape((self.m, self.n)) / (self.size - 1)
        return mat[..., np.newaxis].astype(np.float32)  # shape: (m,n,1)

    def get_moves(self):
        idx = self.state.index(self.empty_tile)
        row, col = divmod(idx, self.n)
        moves = []
        if row > 0: moves.append(0)  # 上
        if row < self.m - 1: moves.append(1)  # 下
        if col > 0: moves.append(2)  # 左
        if col < self.n - 1: moves.append(3)  # 右
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

    def step(self, action):
        legal_moves = self.get_moves()
        reward = -0.01
        done = False
        if action not in legal_moves:
            # reward = -0.1
            reward = - math.square(self.m+self.n)
        else:
            idx = self.state.index(self.empty_tile)
            if action == 0: target = idx - self.n
            elif action == 1: target = idx + self.n
            elif action == 2: target = idx - 1
            elif action == 3: target = idx + 1
            before_total_distance = self._calculate_total_distance(self.state)

            self.state[idx], self.state[target] = self.state[target], self.state[idx]

            # 计算所有滑块到正确位置的曼哈顿距离之和
            total_distance = self._calculate_total_distance(self.state)
            # 使用距离之和作为额外奖励（负奖励，因为距离越小越好）
            distance_reward = -0.01 * total_distance  # 缩放因子可根据需要调整
            # if(before_total_distance>total_distance):
            #     distance_reward = 0.06 # * before_total_distance-total_distance
            # else:
            #     distance_reward = -0.03 # * before_total_distance-total_distance
            reward = distance_reward

            if self.state == list(range(self.size)):
                # reward = 1.0
                reward = math.pow(self.m+self.n,4)
                done = True
        return self._get_obs(), reward, done

# ===== DQN Agent =====
class DQNAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        
        # self.optimizer = tf.keras.optimizers.Adam(1e-3)
        # 动态学习率 指数衰减
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2, decay_steps=32, decay_rate=0.995)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # 或者使用余弦退火
        # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        #     initial_learning_rate=1e-5, decay_steps=1000
        # )
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.model = self._build_model()

    def _build_model(self):
        inputs = tf.keras.Input(shape=(None, None, 1))
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
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
        q_masked = [q[a] for a in legal_moves]
        return legal_moves[np.argmax(q_masked)]

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


if __name__ == '__main__':

    # ===== 训练 =====
    episodes = 500
    reward_list = []

    # 初始化绘图
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlim(0, episodes)
    # ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DQN Training Reward")

    agent = DQNAgent(action_size=4)

    env = SlidingPuzzleEnv(3,3)
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        for t in range(80):
            legal = env.get_moves()
            action = agent.act(state, legal)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done: break
        # for i in range(10):
        agent.replay(batch_size=32)
        reward_list.append(total_reward)

        # ===== 实时更新绘图 =====
        line.set_data(range(len(reward_list)), reward_list)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.05)

        if (e+1) % 5 == 0:
            agent.model.save(MODEL_FILE)
            print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    agent.model.save(MODEL_FILE)

    # ===== 绘制奖励曲线 =====
    plt.plot(reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Reward")
    plt.show()
