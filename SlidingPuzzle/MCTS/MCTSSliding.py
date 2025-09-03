import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pickle
import os
import time
from collections import deque
import random

class MN_Puzzle:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.size = m * n
        self.reset()
    
    def reset(self, scramble_steps=0):
        # 初始状态
        self.state = np.arange(self.size)
        self.empty_pos = self.size - 1
        
        # 随机打乱指定步数
        for _ in range(scramble_steps):
            actions = self.get_actions()
            if actions:
                action = random.choice(actions)
                self.execute_action(action)
        
        return self.state.copy()
    
    def is_solvable(self):
        # 计算逆序数来判断是否可解
        inversions = 0
        flat_state = self.state.flatten()
        for i in range(self.size):
            for j in range(i + 1, self.size):
                if flat_state[i] != self.size - 1 and flat_state[j] != self.size - 1 and flat_state[i] > flat_state[j]:
                    inversions += 1
        
        # 对于m*n拼图，当m为奇数时，逆序数需为偶数；当m为偶数时，逆序数加上空位行数需为奇数
        if self.m % 2 == 1:
            return inversions % 2 == 0
        else:
            empty_row = self.empty_pos // self.n
            return (inversions + empty_row) % 2 == 1
    
    def get_actions(self):
        actions = []
        empty_row, empty_col = self.empty_pos // self.n, self.empty_pos % self.n
        
        if empty_row > 0:
            actions.append('up')
        if empty_row < self.m - 1:
            actions.append('down')
        if empty_col > 0:
            actions.append('left')
        if empty_col < self.n - 1:
            actions.append('right')
            
        return actions
    def negative_action(self,action):
        m = {
            'up':'down',
            'down':'up',
            'left':'right',
            'right':'left',
        }
        return m.get(action)
    
    def execute_action(self, action):
        empty_row, empty_col = self.empty_pos // self.n, self.empty_pos % self.n
        
        if action == 'up' and empty_row > 0:
            swap_pos = (empty_row - 1) * self.n + empty_col
        elif action == 'down' and empty_row < self.m - 1:
            swap_pos = (empty_row + 1) * self.n + empty_col
        elif action == 'left' and empty_col > 0:
            swap_pos = empty_row * self.n + (empty_col - 1)
        elif action == 'right' and empty_col < self.n - 1:
            swap_pos = empty_row * self.n + (empty_col + 1)
        else:
            return False, self.state.copy()
        
        # 交换空位和选中的位置
        self.state[self.empty_pos], self.state[swap_pos] = self.state[swap_pos], self.state[self.empty_pos]
        self.empty_pos = swap_pos
        return True, self.state.copy()
    
    def is_solved(self):
        return np.array_equal(self.state, np.arange(self.size))
    
    def get_state(self):
        return self.state.copy()

class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior=0):
        self.state = state if state is not None else parent.state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.is_expanded = False
    
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def ucb_score(self, exploration_weight=1.0):
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.value()
        exploration = exploration_weight * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        # exploration = exploration_weight * self.prior * np.sqrt(np.log(self.parent.visit_count + 1) / (self.visit_count + 1e-6))
        return exploitation + exploration
    
    def select_child(self):
        return max(self.children.values(), key=lambda child: child.ucb_score())

    def expand(self, action_probs):
        for action, prob in action_probs.items():
            if action not in self.children:
                # 创建新节点
                self.children[action] = MCTSNode(
                    state=None,  # 将在模拟时设置
                    parent=self,
                    action=action,
                    prior=prob
                )
        self.is_expanded = True
    
    def __repr__(self):
        return f"MCTSNode(N={self.visit_count}, V={self.value():.3f})"

class PuzzleNet:
    def __init__(self, m, n, learning_rate=0.001):
        self.m = m
        self.n = n
        self.size = m * n
        self.model = self.build_model()
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate, decay_steps=1000, decay_rate=0.9)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    
    def build_model(self):
        # 输入是拼图状态 (m*n的一维向量)
        inputs = layers.Input(shape=(self.size,))
        
        # 将状态转换为one-hot编码
        x = layers.Lambda(lambda x: tf.one_hot(tf.cast(x, tf.int32), self.size))(inputs)
        x = layers.Reshape((self.m, self.n, self.size))(x)
        
        # 卷积层
        x = layers.Conv2D(4096, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # 全局平均池化
        x = layers.GlobalAveragePooling2D()(x)
        
        # 策略头 - 预测每个动作的概率
        policy = layers.Dense(128, activation='relu')(x)
        policy = layers.Dense(4, activation='softmax', name='policy')(policy)
        
        # 价值头 - 预测当前状态的价值
        value = layers.Dense(128, activation='relu')(x)
        value = layers.Dense(1, activation='tanh', name='value')(value)
        
        model = models.Model(inputs=inputs, outputs=[policy, value])
        return model
    
    def predict(self, state):
        # 将状态转换为模型输入格式
        state_input = np.array(state).reshape(1, -1)
        policy_logits, value = self.model(state_input, training=False)
        
        # 转换为概率分布
        action_probs = tf.nn.softmax(policy_logits).numpy()[0]
        value = value.numpy()[0][0]
        
        # 将动作索引映射为动作名称
        action_names = ['up', 'down', 'left', 'right']
        action_dict = {action_names[i]: action_probs[i] for i in range(4)}
        
        return action_dict, value
    
    def train(self, states, target_policies, target_values):
        with tf.GradientTape() as tape:
            policy_logits, values = self.model(states, training=True)
            
            # 策略损失
            policy_loss = tf.keras.losses.categorical_crossentropy(
                target_policies, policy_logits
            )
            
            # 价值损失
            value_loss = tf.keras.losses.mean_squared_error(
                target_values, tf.reshape(values, [-1])
            )
            
            # 总损失
            total_loss = tf.reduce_mean(policy_loss + value_loss)
        
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss.numpy()

class MCTSAgent:
    def __init__(self, m, n, num_simulations=100, exploration_weight=1.0):
        self.m = m
        self.n = n
        self.size = m * n
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.puzzle_net = PuzzleNet(m, n)
        self.env = MN_Puzzle(m, n)

    def mcts_search(self, root_state):
        root = MCTSNode(root_state)
        
        # # 创建临时环境获取有效动作
        # temp_env = MN_Puzzle(self.m, self.n)
        # temp_env.state = root_state.copy()
        # temp_env.empty_pos = np.where(root_state == self.size - 1)[0][0]
        # valid_actions = temp_env.get_actions()
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # 选择阶段
            while node.is_expanded and node.children:
                node = node.select_child()
                search_path.append(node)
            
            # 扩展阶段
            if not node.is_expanded:
                # 使用神经网络预测动作概率和状态价值
                action_probs, value = self.puzzle_net.predict(node.state)
                node.expand(action_probs)
            else:
                # 随机 rollout 或使用默认策略
                value = self.rollout(node.state)
            
            # 回溯更新
            self.backpropagate(search_path, value)
        
        # # 只从有效动作中选择
        # valid_children = {a: root.children[a] for a in root.children.keys() if a in valid_actions}
        
        # if valid_children:
        #     action = max(valid_children.keys(), key=lambda a: valid_children[a].visit_count)
        # else:
        #     # 备用方案：随机选择有效动作
        #     action = random.choice(valid_actions) if valid_actions else 'up'
        
        action = max(root.children.keys(), key=lambda a: root.children[a].visit_count)
        
        return action, root

    def rollout(self, state):
        # 创建一个临时环境进行随机模拟
        temp_env = MN_Puzzle(self.m, self.n)
        temp_env.state = state.copy()
        temp_env.empty_pos = np.where(state == self.size - 1)[0][0]
        
        max_steps = 50  # 限制最大步数
        for step in range(max_steps):
            if temp_env.is_solved():
                return 1.0  # 成功解决
            
            actions = temp_env.get_actions()
            if not actions:
                return -1.0  # 无可用动作
            
            action = random.choice(actions)
            valid, _ = temp_env.execute_action(action)
            
            if not valid:
                return -1.0  # 无效动作
        
        return 0.0  # 未能在最大步数内解决
    
    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.visit_count += 1
            node.total_value += value
            value = -value  # 交替视角
    
    def self_play(self, num_games=1, scramble_steps=0, max_moves = 150):
        training_data = []
        
        invalid_action_cnt = 0
        for game in range(num_games):
            state = self.env.reset(scramble_steps)
            game_history = []

            # 检查初始状态是否已解决
            if self.env.is_solved():
                print(f"游戏 {game+1} 初始状态已解决")
                continue
            
            # 防止无限循环
            for move in range(max_moves):
                action, root = self.mcts_search(state)
                if action is None:  # 检查是否返回有效动作
                    print("无有效动作，重置环境")
                    break
                
                # 保存训练数据
                action_probs = np.zeros(4)
                action_names = ['up', 'down', 'left', 'right']
                for a, child in root.children.items():
                    idx = action_names.index(a)
                    action_probs[idx] = child.visit_count
                action_probs /= np.sum(action_probs)
                
                game_history.append((state, action_probs))
                
                # 执行动作
                valid, next_state = self.env.execute_action(action)
                if not valid:
                    # print(f"无效动作: {action}，重置环境")
                    invalid_action_cnt +=1
                    break
                
                state = next_state
                
                if self.env.is_solved():
                    print(f"游戏 {game+1} 在 {move+1} 步内解决"+ (f',无效动作{invalid_action_cnt}' if invalid_action_cnt else ''))
                    # 为每一步分配最终结果作为价值目标
                    for i, (s, p) in enumerate(game_history):
                        # 越接近解决，价值越高
                        value_target = 1.0 - 0.5 * (i / len(game_history))
                        training_data.append((s, p, value_target))
                    break
            else:
                print(f"游戏 {game+1} 未能在 {max_moves} 步内解决"+ (f',无效动作{invalid_action_cnt}' if invalid_action_cnt else ''))
                # 为每一步分配负价值
                for s, p in game_history:
                    training_data.append((s, p, -1.0))
        
        return training_data
    

    def train(self, num_iterations=100, num_self_play_games=10, batch_size=32, 
              pre_train_steps=500, loss_threshold=0.1):
        losses = []
        success_rates = []
        replay_buffer = deque(maxlen=10000)
        
        # 创建实时图表
        plt.ion()  # 开启交互模式
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        loss_line, = ax1.plot([], [], 'b-')
        success_line, = ax2.plot([], [], 'r-')
        
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        ax2.set_title('Success Rate')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True)
        
        if hasattr( self, 'pretrain'):
            total_distance = self.pretrain(batch_size,pre_train_steps,loss_threshold)
            replay_buffer.extend(total_distance)
        
        # 2. 主训练阶段：自我对弈学习
        print("开始主训练阶段...")
        max_scramble_steps = 45  # 最大打乱步数
        step_increment = max(1, max_scramble_steps // (num_iterations/4))

        # for iteration in range(num_iterations):
        iteration = 0
        iteration_cnt = 0
        while iteration < num_iterations:
            iteration_cnt += 1
            # 计算当前打乱步数
            scramble_steps = max(3,min(iteration * step_increment, max_scramble_steps))
            print(f"迭代 {iteration+1}/{num_iterations}, 打乱步数: {scramble_steps}, 第 {iteration_cnt/1000:.2f}K 次")
            
            # 自我对弈生成数据
            max_move = min(150,scramble_steps*5)
            training_data = self.self_play(num_self_play_games, scramble_steps,max_move)
            replay_buffer.extend(training_data)
            
            if len(replay_buffer) < batch_size:
                continue
            
            # 从回放缓冲区中采样进行训练
            batch = random.sample(replay_buffer, batch_size)
            states = np.array([data[0] for data in batch])
            target_policies = np.array([data[1] for data in batch])
            target_values = np.array([data[2] for data in batch])
            
            # 训练神经网络
            loss = self.puzzle_net.train(states, target_policies, target_values)
            losses.append(loss)
            
            # 每5次迭代测试一次性能
            if iteration_cnt % 5 == 0:
                success_rate = self.test_performance(10, scramble_steps,max_move)  # 测试5局
                if(success_rate>=90):
                    iteration +=1
                    print(f'成功率提高 next iteration {iteration}')
                success_rates.append(success_rate)
                
                # 更新实时图表
                loss_line.set_data(range(len(losses)), losses)
                ax1.relim()
                ax1.autoscale_view()
                
                success_line.set_data(range(len(success_rates)), success_rates)
                ax2.relim()
                ax2.autoscale_view()
                plt.pause(0.05)
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.1)  # 短暂暂停以更新图表
            
            print(f"损失: {loss:.4f}")
            
            # 保存模型
            if iteration_cnt % 10 == 0:
                self.save_model(f"model_iteration_{iteration+1}")
        
        plt.ioff()  # 关闭交互模式
        plt.savefig('training_progress.png')
        plt.show()
        
        return losses, success_rates
    
    def save_model(self, filepath):
        self.puzzle_net.model.save_weights(f"{filepath}.h5")
        print(f"模型已保存到 {filepath}.h5")
    
    def load_model(self, filepath):
        self.puzzle_net.model.load_weights(f"{filepath}.h5")
        print(f"模型已从 {filepath}.h5 加载")

    # 添加测试性能方法
    def test_performance(self, num_tests=5, scramble_steps=0, max_moves = 150):
        successes = 0
        for i in range(num_tests):
            state = self.env.reset(scramble_steps)
            for step in range(max_moves):  # 最多100步
                action, _ = self.mcts_search(state)
                valid, next_state = self.env.execute_action(action)
                if not valid:
                    break
                state = next_state
                if self.env.is_solved():
                    successes += 1
                    break
        success_rate = (successes / num_tests) * 100
        print(f"测试成功率: {success_rate:.1f}% (打乱步数: {scramble_steps})")
        return success_rate    # 修改保存训练状态方法
    
    def save_training_state(self, losses, success_rates, iteration):
        training_state = {
            'losses': losses,
            'success_rates': success_rates,
            'iteration': iteration,
            'm': self.m,
            'n': self.n
        }
        with open(f'training_state_{iteration}.pkl', 'wb') as f:
            pickle.dump(training_state, f)
        print(f"训练状态已保存到 training_state_{iteration}.pkl")
    
    def load_training_state(self, filepath):
        with open(filepath, 'rb') as f:
            training_state = pickle.load(f)
        
        losses = training_state['losses']
        iteration = training_state['iteration']
        return losses, iteration
    
    def plot_training_progress(self, losses):
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('训练损失')
        plt.xlabel('迭代')
        plt.ylabel('损失')
        plt.grid(True)
        plt.savefig('training_loss.png')
        plt.close()


class MCTSAgent_1 (MCTSAgent):
    def __init__(self, m, n, num_simulations=100, exploration_weight=1):
        print('** MCTSAgent_1 **')
        super().__init__(m, n, num_simulations, exploration_weight)
    def select_child(self):
        # 添加一些随机性
        if random.random() < 1.1:  # 10%概率随机选择
            return random.choice(list(self.children.values()))
        else:
            return max(self.children.values(), key=lambda child: child.ucb_score())    
    
    def mcts_search(self, root_state):
        # 检查是否已经解决
        temp_env = MN_Puzzle(self.m, self.n)
        temp_env.state = root_state.copy()
        temp_env.empty_pos = np.where(root_state == self.size - 1)[0][0]
        
        if temp_env.is_solved():
            return None, None  # 已经解决，无需动作

        root = MCTSNode(root_state)
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # 选择阶段
            while node.is_expanded and node.children:
                # 只从有效动作中选择
                valid_children = {}
                for action, child in node.children.items():
                    # 检查动作是否有效
                    temp_env.state = node.state.copy()
                    temp_env.empty_pos = np.where(node.state == self.size - 1)[0][0]
                    if action in temp_env.get_actions():
                        valid_children[action] = child
                
                if not valid_children:
                    break
                    
                # 选择UCB分数最高的有效子节点
                node = max(valid_children.values(), key=lambda child: child.ucb_score(self.exploration_weight))
                search_path.append(node)
            
            # 检查当前节点是否已解决
            temp_env.state = node.state.copy()
            temp_env.empty_pos = np.where(node.state == self.size - 1)[0][0]
            if temp_env.is_solved():
                value = 1.0  # 已解决，给予最高奖励
            else:
                # 扩展阶段
                if not node.is_expanded:
                    # 使用神经网络预测动作概率和状态价值
                    action_probs, value = self.puzzle_net.predict(node.state)
                    
                    # 过滤无效动作
                    temp_env.state = node.state.copy()
                    temp_env.empty_pos = np.where(node.state == self.size - 1)[0][0]
                    valid_actions = temp_env.get_actions()
                    
                    filtered_action_probs = {}
                    for action, prob in action_probs.items():
                        if action in valid_actions:
                            filtered_action_probs[action] = prob
                    
                    # 重新归一化概率
                    if filtered_action_probs:
                        total_prob = sum(filtered_action_probs.values())
                        for action in filtered_action_probs:
                            filtered_action_probs[action] /= total_prob
                        node.expand(filtered_action_probs)
                    else:
                        # 如果没有有效动作，使用rollout
                        value = self.rollout(node.state)
                else:
                    # 随机rollout
                    value = self.rollout(node.state)
            
            # 回溯更新
            self.backpropagate(search_path, value)
        
        # 只从有效动作中选择
        temp_env.state = root_state.copy()
        temp_env.empty_pos = np.where(root_state == self.size - 1)[0][0]
        valid_actions = temp_env.get_actions()
        
        valid_children = {a: root.children[a] for a in root.children.keys() if a in valid_actions}
        
        if not valid_children:
            return None, root
        
        # 选择访问次数最多的动作
        action = max(valid_children.keys(), key=lambda a: valid_children[a].visit_count)
        return action, root


class MCTSAgent_2 (MCTSAgent):
    def __init__(self, m, n, num_simulations=100, exploration_weight=1):
        print('** MCTSAgent_2 **')
        super().__init__(m, n, num_simulations, exploration_weight)

    def pretrain(self,batch_size=32, 
              pre_train_steps=500, loss_threshold=0.1):
        # 1. 预训练阶段：生成打乱数据进行学习
        print("开始预训练阶段...")
        print(f'生成数据 {pre_train_steps}')
        # pre_train_data = self.generate_pre_train_data(pre_train_steps)
        pre_train_data = self.generate_pre_train_data(pre_train_steps,10)
        
        # 预训练直到损失低于阈值
        pre_train_losses = []
        current_loss = float('inf')
        filter_loss = current_loss
        frate = 0.9
        pre_train_iter = 0

        print('开始训练')
        while filter_loss > loss_threshold and pre_train_iter<pre_train_steps*50:
            if len(pre_train_data) < batch_size:
                continue
                
            batch = random.sample(pre_train_data, batch_size)
            states = np.array([data[0] for data in batch])
            target_policies = np.array([data[1] for data in batch])
            target_values = np.array([data[2] for data in batch])
            
            current_loss = self.puzzle_net.train(states, target_policies, target_values)
            filter_loss = current_loss if filter_loss == float('inf') else filter_loss*frate + current_loss*(1-frate)
            pre_train_losses.append(current_loss)
            pre_train_iter += 1
            
            if pre_train_iter % 40 == 0:
                print(f"预训练迭代 {pre_train_iter}, 损失: {current_loss:.4f} 平均损失: {filter_loss:.4f}")
        
        print(f"预训练完成，最终损失: {current_loss:.4f}")
        return pre_train_data
    def generate_pre_train_data(self,cnt = 100, setp=30):
        """生成预训练数据，基于曼哈顿距离"""
        pre_train_data = []
        temp_env = MN_Puzzle(self.m, self.n)
        action_names = ['up', 'down', 'left', 'right']
        
        # max_distance = 2 * (self.m + self.n - 2)  # 最大可能距离

        # 计算曼哈顿距离作为价值目标
        # manhattan_distance = self.calculate_manhattan_distance(state)
        # value_target_1 = 1.0 - (manhattan_distance / max_distance)  # 曼哈顿距离 归一化到[0,1]
        # value_target_2 = 1.0 - (_ / num_samples) # 步数
        # value_target = 1 - 0.5 * (value_target_1 + value_target_2)/2
        for i in range(cnt):
            temp_env.reset()
            for j in range(setp):
                # 随机打乱
                valid_actions  = temp_env.get_actions()
                action = random.choice(valid_actions )
                _, state = temp_env.execute_action(action)
                
                # 计算价值目标
                value_target = 1.0 - 0.5 * (j / setp) # 步数
                
                action = temp_env.negative_action(action)
                policy_target = np.zeros(4)
                idx = action_names.index(action)
                policy_target[idx] = 1.0
                
                pre_train_data.append((state, policy_target, value_target))
        
        return pre_train_data
    
    def calculate_manhattan_distance(self, state):
        """计算当前状态的曼哈顿距离"""
        total_distance = 0
        for i in range(self.size):
            if state[i] == self.size - 1:  # 空位
                continue
            
            current_row, current_col = i // self.n, i % self.n
            target_row, target_col = state[i] // self.n, state[i] % self.n
            
            total_distance += abs(current_row - target_row) + abs(current_col - target_col)
        
        return total_distance
        