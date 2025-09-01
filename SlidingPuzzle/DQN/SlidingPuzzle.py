import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
import math
from const import SATASET_FILE_NPY,\
MODEL_TEMP_FILE,\
MODEL_FILE,\
EPISODES,\
SIZE_RANGE

# ===== 环境 =====
class SlidingPuzzleEnv:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.size = m * n
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

    def step(self, action):
        legal_moves = self.get_moves()
        reward = -0.01
        done = False
        if action not in legal_moves:
            reward = -0.1
        else:
            idx = self.state.index(self.empty_tile)
            if action == 0: target = idx - self.n
            elif action == 1: target = idx + self.n
            elif action == 2: target = idx - 1
            elif action == 3: target = idx + 1
            self.state[idx], self.state[target] = self.state[target], self.state[idx]

            if self.state == list(range(self.size)):
                reward = 1.0
                done = True
        return self._get_obs(), reward, done

# ===== DQN Agent =====
class DQNAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.memory = {}  # key=(m,n) -> deque
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.lr = 0.001
        self.model = None
        self.dataset = None
        self.batch_size = 32

    def build_model(self, input_shape=[None, None, 1]):
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_size, activation='linear')(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss='mse')
        # 初始化数据集
        self._prepare_dataset()

    def _prepare_dataset(self):
        # 创建数据集生成器
        def data_generator():
            while True:
                if not self.memory:
                    yield np.zeros((1, 1, 1, 1)), np.zeros((1, self.action_size))
                else:
                    key = random.choice(list(self.memory.keys()))
                    mem = self.memory[key]
                    if len(mem) < self.batch_size:
                        batch_size = len(mem)
                    else:
                        batch_size = self.batch_size
                    batch = random.sample(mem, batch_size)
                    states, targets = [], []
                    for s, a, r, s2, done in batch:
                        target = r
                        if not done:
                            q_next = self.model.predict(s2[np.newaxis, ...], verbose=0)[0]
                            target += self.gamma * np.max(q_next)
                        q_current = self.model.predict(s[np.newaxis, ...], verbose=0)[0]
                        q_current[a] = target
                        states.append(s)
                        targets.append(q_current)
                    yield np.array(states), np.array(targets)

        # 创建 TensorFlow Dataset
        self.dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.action_size), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)

    def load_model(self):
        # 加载模型和优化器状态
        if os.path.exists(MODEL_TEMP_FILE):
            print("加载上次中断的模型")
            self.model.load_weights(MODEL_TEMP_FILE)
            self.load_memory()
            self._prepare_dataset()  # 重新准备数据集

    def save_temp_model(self):
        self.save_memory()
        self.model.save(MODEL_TEMP_FILE)
        self.print_memory_info(True)

    def save_model(self):
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        if os.path.exists(MODEL_TEMP_FILE):
            os.rename(MODEL_TEMP_FILE, MODEL_FILE)

    def save_memory(self):
        save_dict = {}
        for key, mem in self.memory.items():
            # mem 是 deque，转换为 list
            mem_list = list(mem)
            # state/next_state shape 不统一，用 dtype=object 保存
            mem_array = np.array(mem_list, dtype=object)
            save_dict[key] = mem_array
        np.save(SATASET_FILE_NPY, save_dict)

    def load_memory(self):
        if not os.path.exists(SATASET_FILE_NPY):
            print(f"memory 文件没找到 {SATASET_FILE_NPY}")
            return {}
        load_dict = np.load(SATASET_FILE_NPY, allow_pickle=True).item()
        print(f"Memory 文件已加载")
        list_dict = {}
        for key, mem in load_dict.items():
            mem_array = mem.tolist()
            list_dict[key] = mem_array
        self.memory = list_dict
        self.print_memory_info()
        return list_dict

    def print_memory_info(self, simple=False):
        if simple:
            cnt = 0
            total = 0
            for key, mem in self.memory.items():
                cnt += 1
                total += len(mem)
            print(f"memory info: keys{len(self.memory.keys())}")
        else:
            print("memory info:")
            for key, mem in self.memory.items():
                print(f"  key:{key}, len:{len(mem)}")

    def remember(self, s, a, r, s2, done):
        key = (s.shape[0], s.shape[1])
        if key not in self.memory:
            self.memory[key] = deque(maxlen=2000)
        self.memory[key].append((s, a, r, s2, done))

    def act(self, state, legal_moves):
        if np.random.rand() < self.epsilon:
            return random.choice(legal_moves)
        q = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
        q_masked = [q[a] for a in legal_moves]
        return legal_moves[np.argmax(q_masked)]

    def replay(self, batch_size=32):
        if not self.memory:
            return
        self.batch_size = batch_size
        # 使用数据集进行训练
        data_iter = iter(self.dataset)
        states, targets = next(data_iter)
        self.model.train_on_batch(states, targets)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ===== 训练 =====
reward_list = []
agent = DQNAgent(action_size=4)

# 初始化绘图
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlim(0, EPISODES)
# ax.set_ylim(-0.2, 1.2)
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.set_title("DQN Training Reward")

max_step_dic = {
    2:4,
    3:31,
    4:80,
    5:208,
    6:400,
}

REPLAY_PERIOD = 30
e = 0
total_reward = 0
# for e in range(EPISODES):
while e < EPISODES * REPLAY_PERIOD:
    # 每个 episode 随机一个尺寸 m×n
    m = random.randint(*SIZE_RANGE)
    n = random.randint(*SIZE_RANGE)
    env = SlidingPuzzleEnv(m, n)
    state = env.reset()

    # 第一次 episode 建立模型
    if agent.model is None:
        agent.build_model()
        agent.load_model()

    max_step = math.floor(max_step_dic.get(max(m,n),400) *1.3)
    for t in range(max_step):
        legal = env.get_moves()
        action = agent.act(state, legal)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done: break

        e+=1
        if e and e % REPLAY_PERIOD ==0:
            agent.replay(batch_size=32)
            reward_list.append(total_reward)

            # ===== 实时更新绘图 =====
            line.set_data(range(len(reward_list)), reward_list)
            ax.relim()
            ax.autoscale_view()
            # sleep(0.002)
            plt.pause(0.05)

            if (e) % (5*REPLAY_PERIOD) == 0:
                agent.save_temp_model()
                print(f"Episode {e//REPLAY_PERIOD+1}/{EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            total_reward = 0

agent.save_model()

# ===== 绘制奖励曲线 =====
plt.ioff()
plt.plot(reward_list)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Reward")
plt.show()
