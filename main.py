from ast import literal_eval
from sumolib import checkBinary
import traci
from collections import deque
import random
from env2 import SumoEnv
from agent import ReplayBuffer, QNet, Agent

def main():
    episodes = 10000
    time_limit = 3600 
    block_veh_rate = 0.5
    intersections = env.intersections
    lane_dict = env.lane_dict
    agents = {}
    #交差点の数だけエージェントインスタンス化
    for intersection in intersections:
        agents[intersection] = Agent(id=intersection,lane_info=lane_dict[intersection])
    print("agents:", agents)
    for episode in range(1,episodes+1):
        done = False
        step = 0
        time = 0
        veh_rate = 0
        env.reset()
        #sumo.get_shape()
        #grid_frag = False
        states = {}
        actions = {}
        rewards = {}
        
        #初期状態
        for id, agent in agents.items(): #.items()でkeyと要素を両方と取得
            states[id] = agent.get_state()
        #優先表示用ポリゴン
        #env.define_polygon()
        while not done:
            #車両生成
            current_time = traci.simulation.getTime()
            veh_rate = random.random() * 10
            veh_num = int(veh_rate)
            for i in range(veh_num):
                if random.random() < block_veh_rate:
                    vtype = "block_car"
                    #print("block")
                else:
                    vtype = "normal_car"
                    #print("normal")
                env.make_vehicle(f"vehicle_{step}_{i}", env.make_random_route(step, i), current_time, vtype)
            
            for id, agent in agents.items():
                actions[id] = agent.get_action(states[id])
            
            rewards, next_state, done = env.step(actions)
            state = next_state
            print(f"action:{actions}")
            print(f"reward:{rewards}")
            if current_time >= time_limit:
                done = True
            step += 1
        traci.close()


if __name__ == "__main__":
    env = SumoEnv()
    main()