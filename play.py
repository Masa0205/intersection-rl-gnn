from ast import literal_eval
from sumolib import checkBinary
import traci
from collections import deque
from datetime import datetime
from env2 import SumoEnv
from agent import Agent
from util import r_graph
import random
def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    episodes = 1
    rollout_interval = 50
    print_interval = 10
    intersections = env.intersections
    lane_dict = env.lane_dict
    agents = {}
    r_total = {i: [] for i in intersections}
    loss_total = {i: [] for i in intersections}
    t_total = 0
    print_r = {}
    print_loss = {}
    #交差点の数だけエージェントインスタンス化
    for intersection in intersections:
        agents[intersection] = Agent(id=intersection,lane_info=lane_dict[intersection])
    #パラメータ読み込み
    for id, agent in agents.items():
        agent.load(id, date)
    print("agents:", agents)
    for episode in range(episodes):
        r_sum = {i: 0.0 for i in intersections}
        loss_sum = {i: 0.0 for i in intersections}
        states = {}
        next_states = {}
        actions = {i: random.randint(0,3) for i in intersections}
        rewards = {}
        train_step = 0
        done = False
        step = 0
        vehicle_num = 1000
        env.reset(is_gui)
        env.get_internal()
        env.get_pair()
        #sumo.get_shape()
        #grid_frag = False
        
        
        #初期状態
        for intersection in intersections: #.items()でkeyと要素を両方と取得
            states[intersection] = env.get_state(intersection, actions[intersection])
        #優先表示用ポリゴン
        #env.define_polygon()
        for i in range(vehicle_num):
                depart_time = random.randint(0,500)
                env.make_vehicle(f"vehicle_{i}", depart_time, i)
        while not done:
            
            for id, agent in agents.items():
                actions[id] = agent.get_action(states[id], play=True)
            print(f"actions: {actions}")
            rewards, next_states, done = env.step(actions, rewards, next_states)
            for intersection in intersections:
                r_sum[intersection] += rewards[intersection]
                states[intersection] = next_states[intersection]
            if done == True:
                t_total += traci.simulation.getTime()
            
            step += 1
        for i in intersections:
            r_total[i].append(r_sum[i] / step)
        #報酬はエピソード平均、損失はエピソード終了時に算出した平均、時間は10エピソード平均
        if episode % print_interval == 0:
            for i in intersections:
                print_r[i] = r_total[i][-1]
            print(f"eps: {episode} r: {print_r} t: {t_total / print_interval}")
            t_total = 0
        traci.close()
    r_graph(r_total, timestamp)

if __name__ == "__main__":
    is_gui = True
    env = SumoEnv()
    date = input("date?: ")
    main()