import numpy as np
import tensorflow as tf
from MCTSSliding import MCTSAgent,MCTSAgent_1,MCTSAgent_2

def main():
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 创建3x3拼图代理
    # agent = MCTSAgent(3, 3, num_simulations=50)
    # agent = MCTSAgent_1(3, 3, num_simulations=50)
    agent = MCTSAgent_2(3, 3, num_simulations=50)

    # # 检查是否有之前的训练状态可以加载
    # agent.load_model('model_iteration_3')
    
    losses = []
    
    # 训练代理
    try:
        new_losses = agent.train(
            num_iterations=100,
            num_self_play_games=10,
            batch_size=64
        )
        losses.extend(new_losses)
        
        # 保存最终模型和训练状态
        agent.save_model("model_final")
        
    except KeyboardInterrupt:
        print("训练被中断，保存当前状态...")
        agent.save_model("model_interrupted")
    
    # 测试训练好的模型
    test_agent_performance(agent)

def test_agent_performance(agent:MCTSAgent):
    """测试代理性能"""
    print("测试代理性能...")
    successes = 0
    total_steps = 0
    
    for i in range(10):
        state = agent.env.reset()
        steps = 0
        max_steps = 100
        
        for step in range(max_steps):
            action, _ = agent.mcts_search(state)
            valid, next_state = agent.env.execute_action(action)
            
            if not valid:
                print(f"测试 {i+1}: 无效动作 {action}")
                break
            
            state = next_state
            steps += 1
            
            if agent.env.is_solved():
                successes += 1
                total_steps += steps
                print(f"测试 {i+1}: 在 {steps} 步内解决")
                break
        else:
            print(f"测试 {i+1}: 未能在 {max_steps} 步内解决")
    
    print(f"成功率: {successes}/10")
    if successes > 0:
        print(f"平均步数: {total_steps/successes:.2f}")

if __name__ == "__main__":
    main()