import numpy as np
import random
import tensorflow as tf
from SlidingPuzzle_0 import SlidingPuzzleEnv, MODEL_FILE,SlidingPuzzleDetector
import msvcrt


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
            
            # # 最后显示完成状态
            # if done:
            #     clear_previous_lines()
            #     print(f"\n步骤 {len(solution_path)}/{len(solution_path)}: 完成!")
            #     print("最终状态:")
            #     print(detector.get_state_string(env_demo.state))
            #     print("拼图完成!")                
            #     input("按回车继续下一步...")
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