# experiments/run_tabular_baseline.py


from envs.drone_grid_env import DroneGridEnv
from agents.tabular_qlearning import TabularQLearningAgent


def run_episode(env, agent, train: bool = True):
    """
    Run one episode.
    Returns: total_reward, success (bool)
    """
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        if train:
            agent.update(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

    # Success = ended at goal
    success = (tuple(env.agent_pos) == env.goal)
    return total_reward, success


def main():
    # 1) Create environment (small grid)
    env = DroneGridEnv(grid_size=5, max_steps=50)

    # 2) Create agent
    agent = TabularQLearningAgent(
        grid_size=5,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.2,   # higher exploration during training is ok
    )

    # 3) Train
    train_episodes = 200
    print(f"\n[TRAIN] episodes={train_episodes}")
    for ep in range(1, train_episodes + 1):
        ep_reward, success = run_episode(env, agent, train=True)

        # Print every 20 episodes (keeps output readable)
        if ep % 20 == 0:
            status = "SUCCESS" if success else "FAIL"
            print(f"Episode {ep:03d} | reward={ep_reward:7.2f} | {status}")

    # 4) Evaluate (turn off exploration)
    agent.epsilon = 0.0
    eval_episodes = 20
    successes = 0

    print(f"\n[EVAL] episodes={eval_episodes} (epsilon=0)")
    for ep in range(1, eval_episodes + 1):
        ep_reward, success = run_episode(env, agent, train=False)
        successes += int(success)
        status = "SUCCESS" if success else "FAIL"
        print(f"Eval {ep:02d} | reward={ep_reward:7.2f} | {status}")

    success_rate = successes / eval_episodes
    print(f"\nDone. Success rate = {successes}/{eval_episodes} = {success_rate:.2f}")


if __name__ == "__main__":
    main()
