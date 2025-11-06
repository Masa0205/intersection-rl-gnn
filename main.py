from ast import literal_eval
from sumolib import checkBinary
import traci
from collections import deque
from datetime import datetime
from env2 import SumoEnv
from agent import Agent
from util import r_graph, loss_graph
def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    episodes = 10000
    rollout_interval = 50
    print_interval = 10
    intersections = env.intersections
    lane_dict = env.lane_dict
    agents = {}
    states = {}
    next_states = {}
    actions = {}
    rewards = {}
    r_total = {i: [] for i in intersections}
    loss_total = {i: [] for i in intersections}
    t_total = 0
    print_r = {}
    print_loss = {}
    #交差点の数だけエージェントインスタンス化
    for intersection in intersections:
        agents[intersection] = Agent(id=intersection,lane_info=lane_dict[intersection])
    print("agents:", agents)
    for episode in range(episodes):
        r_sum = {i: 0.0 for i in intersections}
        loss_sum = {i: 0.0 for i in intersections}
        train_step = 0
        done = False
        step = 0
        env.reset()
        #sumo.get_shape()
        #grid_frag = False
        
        
        #初期状態
        for intersection in intersections: #.items()でkeyと要素を両方と取得
            states[intersection] = env.get_state(intersection)
        #優先表示用ポリゴン
        #env.define_polygon()
        while not done:

            for id, agent in agents.items():
                actions[id] = agent.get_action(states[id])

            rewards, next_states, done = env.step(actions, rewards, next_states)
            for id, agent in agents.items():
                agent.buffer.rewards.append(rewards[id])
                r_sum[id] += rewards[id]
                agent.buffer.dones.append(done)
                states[id] = next_states[id]
            step += 1
            if step % rollout_interval == 0:
                train_step += 1
                for i, agent in agents.items():
                    loss = agent.train()
                    loss_sum[i] += loss
        
        t_total += step*10
        for i in intersections:
            r_total[i].append(r_sum[i] / step)
            if train_step == 0:
                loss_total[i].append(loss_sum[i])
            else:
                loss_total[i].append(loss_sum[i] / train_step)

        if episode % print_interval == 0:
            for i in intersections:
                print_r[i] = r_total[i][-1]
                print_loss[i] = loss_total[i][-1]
            print(f"eps: {episode} r: {print_r} loss: {print_loss} t: {t_total / print_interval}")
            t_total = 0
        traci.close()
    r_graph(r_total, timestamp)
    loss_graph(loss_total, timestamp)
    agent.save(timestamp)

if __name__ == "__main__":
    env = SumoEnv()
    main()