import sumolib
from sumolib import checkBinary
import traci
import random
import os
import sys
import numpy as np

class SumoEnv:
    def __init__(self):
        #SUMO check
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            print(os.environ['SUMO_HOME'])
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")
        
        self.roots = {"-gneE64": ["gneE65","gneE61"],
                      "gneE17": ["gneE61","gneE65"],
                      "gneE4": ["gneE60","gneE63"],
                      "-gneE60": ["gneE60","gneE63"],
                      "-gneE65": ["gneE64", "-gneE17"],
                      "-gneE61": ["gneE64","-gneE17"],
                      "-gneE62": ["-gneE4","gneE60"],
                      "-gneE63": ["-gneE4","gneE60"]}

        self.SPEED = 5
        self.DISTANCE = self.SPEED * 10
        self.step_num = 10
        self.net = sumolib.net.readNet('data/intersection.net.xml')
        self.shape = []
        self.lane_dict = {}
        self.polygon_dict = []
        self.time_limit = 3600 
        self.block_veh_rate = 0.5
        self.priority = {}
        self.intersections = ["gneJ41", "gneJ4", "gneJ5", "gneJ42"]
        #各交差点の侵入、流出レーン辞書定義
        self.lane_dict = {}
        for intersection in self.intersections:
            incoming = []
            outgoing = []
            for edge in self.net.getNode(intersection).getIncoming():
                edge_id = edge.getID()
                incoming.append(edge_id)
            for edge in self.net.getNode(intersection).getOutgoing():
                edge_id = edge.getID()
                outgoing.append(edge_id)
            self.lane_dict[intersection] = {
                "incoming": incoming,
                "outgoing": outgoing
            }
        print("lane_dict:", self.lane_dict)
        #初期優先権
        for intersection in self.intersections:
            priority_edge = random.choice(self.lane_dict[intersection]["incoming"])
            self.priority[intersection] = f"{priority_edge}_0"
        
        self.detect_dict = {
            "gneJ41": ["e2Detector_gneE64_0_0", "e2Detector_-gneE56_0_2", "e2Detector_gneE57_0_3", "e2Detector_gneE63_0_1"],
            "gneJ4": ["e2Detector_-gneE17_0_14", "e2Detector_-gneE4_0_13", "e2Detector_gneE5_0_12", "e2Detector_gneE56_0_15"],
            "gneJ5":["e2Detector_-gneE5_0_9", "e2Detector_gneE60_0_10", "e2Detector_gneE65_0_11", "e2Detector_-gneE58_0_8"],
            "gneJ42": ["e2Detector_-gneE57_0_4", "e2Detector_gneE58_0_7", "e2Detector_gneE61_0_6", "e2Detector_gneE62_0_5"]
        }
        
       
    def define_polygon(self):
        for lane in traci.lane.getIDList():
            if ":" in lane:
                continue
            try:
                lane_shape = traci.lane.getShape(lane)
                #print(f"{lane}'s shape: {lane_shape}")
                if lane in traci.polygon.getIDList():
                    traci.polygon.remove(lane)
                traci.polygon.add(
                    lane,
                    lane_shape,
                    (0, 0, 0, 0),
                    fill=True,
                    layer=50,
                    polygonType="lane_highlight"
                )
                #print(lane)
            except traci.TraCIException as e:
                print(e)
    
    def reset(self, is_gui=False):
        if is_gui:
            sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo-gui')
        else:
            sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo')
        sumoCmd = [sumoBinary, "-c", "data/intersection.sumocfg",  "--time-to-teleport", "-1"]
        #,"--ignore-junction-blocker", "0"
        traci.start(sumoCmd)

    

    def make_vehicle(self, vehID, routeID, depart_time, vtype):
        traci.vehicle.add(vehID, routeID, depart=depart_time, typeID=vtype, departLane="best")
        traci.vehicle.setSpeed(vehID, self.SPEED)
        traci.vehicle.setMaxSpeed(vehID, self.SPEED)
    
    def make_random_route(self, step, num):
        depart = random.choice(list(self.roots.keys()))
        arrive = random.choice(self.roots[depart])
        traci.route.add(f"random_route_{step}_{num}", [depart, arrive])   
        return f"random_route_{step}_{num}"
    
    def get_shape(self, intersection):
        shape = []
        shape = traci.junction.getShape(intersection)
        #print(f"intersection's shape={shape}")
        return shape
    

    def set_speed(self):
        MARGIN = 3
        for lane_id in traci.lane.getIDList():
            vehicles = []
            if ":" in lane_id:
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                for veh in vehicles:
                    traci.vehicle.setSpeed(veh, self.SPEED)
            #traci.polygon.setColor(lane_id,(0,0,0,0))

            elif lane_id in list(self.priority.values()):
                #print(f"priorityLane:{lane_id}")
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                if not vehicles:
                    continue
                last_veh = vehicles[-1]
                traci.vehicle.setSpeed(last_veh, self.SPEED)
                traci.vehicle.setColor(last_veh, (255,0,0))
                """
                try:
                    print(f"set polygon color {lane_id}")
                    traci.polygon.setColor(lane_id,(0,255,0,150))
                    traci.polygon.setFilled(lane_id, True)
                except traci.TraCIException as e:
                    print(e)
                """
            else:
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                if not vehicles:
                    continue
                last_veh = vehicles[-1]
                dist = traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(last_veh)
                if dist <= MARGIN:
                    traci.vehicle.setSpeed(last_veh,0)
                    traci.vehicle.setColor(last_veh, (0,255,0))

    def get_state(self, intersection):
        SPEED = 5
        #各車線の特徴量ベクトル
        #流入車線：[車両密度,平均待ち時間,平均速度]
        #流出車線：[車両密度,平均待ち時間,平均速度]
        obs_len = 50
        
        
        jx, jy = traci.junction.getPosition(intersection)
        state = []
        in_vehicles = []
        out_target_veh = []
        #流入車線
        for edgeID in self.lane_dict[intersection]["incoming"]:
            in_vehicles = traci.edge.getLastStepVehicleIDs(edgeID)
            in_target_veh = []
            for veh in in_vehicles:
                
                x, y = traci.vehicle.getPosition(veh)
                dis = np.sqrt((x - jx)**2 + (y - jy)**2)
                if dis <= obs_len:
                    in_target_veh.append(veh)
            if len(in_target_veh) > 0:
                #車両密度
                state.append(len(in_target_veh) / obs_len)

                vel = 0
                wait = 0
                for veh in in_target_veh:
                    vel += traci.vehicle.getSpeed(veh)
                    wait += traci.vehicle.getWaitingTime(veh)
                #平均待ち時間
                state.append(wait / len(in_target_veh))
                #平均速度
                state.append((vel / len(in_target_veh)) / SPEED)
            #print(f"incoming-{i}-state:{in_state[i]}")
            else:
                for _ in range(3):
                    state.append(0)
        
        #流出車線
        for edgeID in self.lane_dict[intersection]["outgoing"]:
            out_vehicles = traci.edge.getLastStepVehicleIDs(edgeID)
            out_target_veh = []
            for veh in out_vehicles:
                
                x, y = traci.vehicle.getPosition(veh)
                dis = np.sqrt((x - jx)**2 + (y - jy)**2)
                if dis <= obs_len:
                    out_target_veh.append(veh)
            if len(out_target_veh) > 0:
                #車両密度
                state.append(len(out_target_veh) / obs_len)

                vel = 0
                wait = 0
                for veh in out_target_veh:
                    vel += traci.vehicle.getSpeed(veh)
                    wait += traci.vehicle.getWaitingTime(veh)
                #平均待ち時間
                state.append(wait / len(out_target_veh))
                #平均速度
                state.append((vel / len(out_target_veh)) / SPEED)
            #print(f"outgoing-{i}-state:{out_state[i]}")
            else:
                for _ in range(3):
                    state.append(0)
        #print(f"{intersection}_state: {state}")

        return state

    def done_check(self, current_time):
        done = False
        vehicles = []
        vehicles = traci.vehicle.getIDList()
        if len(vehicles) == 0:
            return done
        for veh in vehicles:
            if not traci.vehicle.getSpeed(veh) < 0.1:
                if current_time >= 3600:
                    done = True
                return done
        done = True
        return done 

    def step(self, actions, rewards, states):
        for intersection in self.intersections:
            priority_edge = self.lane_dict[intersection]["incoming"][actions[intersection]]
            priority_lane = f"{priority_edge}_0"
            self.priority[intersection] = priority_lane
        #print("pritority", self.priority)
        self.set_speed()
        reward_sum = {i: 0.0 for i in self.intersections}
        state_sum = {i: np.zeros(len(self.get_state(i))) for i in self.intersections}
        for i in range(self.step_num):
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            veh_rate = random.random() * 5
            veh_num = int(veh_rate)
            for i in range(veh_num):
                if random.random() < self.block_veh_rate:
                    vtype = "block_car"
                    #print("block")
                else:
                    vtype = "normal_car"
                    #print("normal")
                self.make_vehicle(f"vehicle_{current_time}_{i}", self.make_random_route(current_time, i), current_time, vtype)

            for intersection in self.intersections:
                throuput = 0
                for detector in self.detect_dict[intersection]:
                    throuput += traci.lanearea.getLastStepVehicleNumber(detector)
                reward_sum[intersection] += throuput
                state_sum[intersection] += self.get_state(intersection)
        for i in self.intersections:
            rewards[i] = reward_sum[i] / self.step_num
            states[i] = state_sum[i] / self.step_num
        #print("rewards",rewards)
        #print("states", states)
        done = self.done_check(current_time)
        #print("done", done)
        
        return rewards, states, done
            

       
            
                





        


    def resolve_stuck_vehicles(self):
        """
        レーン間違いで硬直している車両を検出し、
        そのレーンから行ける方向に経路を再設定する関数。
        """
        for veh_id in traci.vehicle.getIDList():
            try:
                # 停止している車両をチェック
                if traci.vehicle.getSpeed(veh_id) < 0.1:
                    current_lane = traci.vehicle.getLaneID(veh_id)
                    route = traci.vehicle.getRoute(veh_id)
                    current_edge = traci.vehicle.getRoadID(veh_id)

                    # 車両の経路がまだ先にあるか確認
                    if current_edge in route:
                        current_edge_index = route.index(current_edge)
                        if current_edge_index < len(route) - 1:
                            next_edge_in_route = route[current_edge_index + 1]

                            # 現在のレーンから行ける接続先（リンク）を取得
                            links = traci.lane.getLinks(current_lane)
                            
                            can_proceed = False
                            possible_next_edges = []
                            for link in links:
                                # リンク先のレーンのエッジIDを取得
                                to_edge = traci.lane.getEdgeID(link[0])
                                possible_next_edges.append(to_edge)
                                if to_edge == next_edge_in_route:
                                    can_proceed = True
                                    break
                            
                            # もし本来の経路に進めない場合 (can_proceed is False)
                            if not can_proceed and possible_next_edges:
                                # --- ここで経路修正 ---
                                new_target_edge = possible_next_edges[0] # とりあえず最初に見つかった行ける道へ
                                traci.vehicle.changeTarget(veh_id, new_target_edge)
                                #print(f"車両 {veh_id} はレーンを間違えたため、経路を {new_target_edge} へ修正しました。")

            except traci.TraCIException:
                # 車両がシミュレーションから離れた場合などのエラーを無視
                continue
            






            