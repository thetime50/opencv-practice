import numpy as np
import random
import tensorflow as tf
from SlidingPuzzle_0 import SlidingPuzzleEnv, MODEL_FILE
import msvcrt

class SlidingPuzzleDetector:
    def __init__(self, m=3, n=3):
        self.env = SlidingPuzzleEnv(m, n)
        self.model = tf.keras.models.load_model(MODEL_FILE)
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

# 测试代码
if __name__ == "__main__":
    detector = SlidingPuzzleDetector(3, 3)
    
    for test_idx in range(10):
        print(f"\n=== 测试 {test_idx + 1}/10 ===")
        
        # 重置环境并随机打乱n步
        env = SlidingPuzzleEnv(3, 3)
        initial_state = env.reset()
        shuffle_steps = random.randint(5, 15)
        
        # 随机打乱
        for _ in range(shuffle_steps):
            legal_moves = env.get_moves()
            action = random.choice(legal_moves)
            env.step(action)
        
        scrambled_state = env.state.copy()
        print("初始打乱状态:")
        print(detector.get_state_string(scrambled_state))
        
        # 使用solve求解
        success, solution_path = detector.solve(scrambled_state, max_steps=30)
        
        if success:
            print(f"求解成功! 需要 {len(solution_path)} 步")
        else:
            print(f"求解失败! 已尝试 {len(solution_path)} 步")
        
        print("解决方案动作序列:", [detector.action_names[action] for action in solution_path])
        
        # 检查用户输入
        print("按 'a' 逐步演示，其他键跳过演示: ")
        user_input = msvcrt.getch().decode().lower()        
        if user_input == 'a':
            print("\n逐步演示:")
            current_state = scrambled_state.copy()
            env_demo = SlidingPuzzleEnv(3, 3)
            env_demo.state = current_state
            
            # 清屏函数
            def clear_previous_lines(lines=7):
                for _ in range(lines):
                    print("\033[F\033[K", end="")  # 光标上移一行并清除该行
            
            for i, action in enumerate(solution_path):
                # 清除之前的输出（假设每次输出约10行）
                if i > 1:
                    clear_previous_lines()
                
                print(f"\n步骤 {i + 1}/{len(solution_path)}: {detector.action_names[action]}")
                print("当前状态:")
                print(detector.get_state_string(env_demo.state))
                print("按a继续下一步...")
                
                # 执行动作
                next_state, reward, done = env_demo.step(action)
                
                if done:
                    print("拼图完成!")
                    break
                
                user_input = msvcrt.getch().decode().lower()  # 等待用户按回车
                if(user_input != 'a'):
                    print("退出动画!")
                    break
            
            # 最后显示完成状态
            if done:
                clear_previous_lines()
                print(f"\n步骤 {len(solution_path)}/{len(solution_path)}: 完成!")
                print("最终状态:")
                print(detector.get_state_string(env_demo.state))
                print("拼图完成!")                
                input("按回车继续下一步...")
        else:
            # 验证解决方案
            final_state = detector.apply_action_sequence(scrambled_state, solution_path)
            if final_state is not None:
                print("验证最终状态:")
                print(detector.get_state_string(final_state))
                if final_state == list(range(9)):
                    print("验证成功: 拼图已解决!")
                else:
                    print("验证失败: 拼图未完全解决")
            else:
                print("验证失败: 动作序列包含非法移动")
        
        print("-" * 40)