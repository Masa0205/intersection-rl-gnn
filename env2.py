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

        self.SPEED = 10
        self.DISTANCE = self.SPEED * 10
        self.step_num = 10
        self.net = sumolib.net.readNet('data/intersection.net.xml')
        self.shape = []
        self.lane_dict = {}
        self.polygon_dict = []
        self.time_limit = 3600 
        self.block_veh_rate = 0.5
        self.obs_len = 30
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
        #print("lane_dict:", self.lane_dict)

        
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
    
    def get_pair(self):
        self.pair_dict = {}
        for intersection, value in self.lane_dict.items():
            self.pair_dict[intersection] = {}
            for edge in value["incoming"]:
                lane_count = traci.edge.getLaneNumber(edge)
                for i in range(lane_count):
                    in_lane_id = f"{edge}_{i}"
                    
                    # 【重要】このレーンからの接続先（リンク）を取得
                    # links = [(connectedLaneID, isOpen, hasPrio, isFoe), ...]
                    links = traci.lane.getLinks(in_lane_id)
                    
                    pair_lanes = []
                    for link in links:
                        out_lane_id = link[0] # 0番目の要素が接続先のレーンID
                        
                        # Uターンを除外したい場合（オプション）
                        # 接続先のエッジが、流入エッジの反対方向なら除外する等の処理が可能
                        # if out_lane_id.startswith(edge): continue 

                        pair_lanes.append(out_lane_id)
                self.pair_dict[intersection][in_lane_id] = pair_lanes
        #print(f"pair_dict: {self.pair_dict}")

       
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
        sumoCmd = [sumoBinary, "-c", "data/intersection.sumocfg", "--time-to-teleport", "-1", "--collision.action", "none"]
        # ,"--time-to-teleport", "-1"   ,"--ignore-junction-blocker", "0"
        traci.start(sumoCmd)

    def get_internal(self):
        lanes = traci.lane.getIDList()
        for intersection in self.lane_dict.keys():
            internals = []
            for lane in lanes:
                if f":{intersection}" in lane:
                    internals.append(lane)
            self.lane_dict[intersection]["internal"] = internals
        #print(f"lane_dict: {self.lane_dict}")


    

    def make_vehicle(self, vehID, depart_time, i):
        """
        if random.random() < self.block_veh_rate:
            vtype = "block_car"
            #print("block")
        else:
            vtype = "normal_car"
            #print("normal")
        """
        vtype = "block_car"
        traci.vehicle.add(vehID, routeID="random_routes", depart=depart_time, typeID=vtype, departLane="best")
        traci.vehicle.setSpeed(vehID, self.SPEED)
        traci.vehicle.setMaxSpeed(vehID, self.SPEED)
    
    def make_random_route(self, num):
        depart = random.choice(list(self.roots.keys()))
        arrive = random.choice(self.roots[depart])
        traci.route.add(f"random_route_{num}", [depart, arrive])
        return f"random_route_{num}"
    
    def get_shape(self, intersection):
        shape = []
        shape = traci.junction.getShape(intersection)
        #print(f"intersection's shape={shape}")
        return shape
    

    def set_speed(self):
        #赤：優先　緑：停止　黄色：SUMO制御車両
        MARGIN = 3
        STOP_THRESHOLD = 20

        for id in traci.lane.getIDList():
            if ":" in id:
                    vehicles = traci.lane.getLastStepVehicleIDs(id)
                    for veh in vehicles:
                        traci.vehicle.setSpeed(veh, self.SPEED)
                        traci.vehicle.setSpeedMode(veh, 7)
                        traci.vehicle.setColor(veh, (255,255,255))
                        #
                #traci.polygon.setColor(lane_id,(0,0,0,0))

        for intersection, priority_lane in self.priority.items():
            for lane in self.lane_dict[intersection]["incoming"]:
                lane_id = f"{lane}_0"
                vehicles = []
                

                if lane_id == priority_lane:
                    #print(f"priorityLane:{lane_id}")
                    vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                    if not vehicles:
                        continue
                    for veh in vehicles:
                        if veh == vehicles[-1]:
                            traci.vehicle.setSpeed(veh, self.SPEED)
                            traci.vehicle.setSpeedMode(veh, -1)
                            traci.vehicle.setColor(veh, (255,0,0))
                        else:
                            traci.vehicle.setSpeedMode(veh, -1)
                            traci.vehicle.setColor(veh, (255,255,0))
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
                    for veh in vehicles:
                        if veh == vehicles[-1]:
                            dist = traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(veh)
                            if dist <= MARGIN:
                                traci.vehicle.setSpeed(veh, 0)
                                traci.vehicle.setColor(veh, (0,255,0))
                            elif dist <= STOP_THRESHOLD:
                                # 停止線に近いがまだ余裕がある -> 徐行で詰める (例: 2m/s)
                                # これをしないと、勢いよく突っ込んできて止まりきれない
                                traci.vehicle.setSpeed(veh, 2.0) 
                                traci.vehicle.setSpeedMode(veh, 1) 
                                traci.vehicle.setColor(veh, (0, 255, 128)) # 薄い緑
                            else:
                                traci.vehicle.setSpeedMode(veh, -1)
                                traci.vehicle.setColor(veh, (255,255,0))
                        else:
                            traci.vehicle.setSpeedMode(veh, -1)
                            traci.vehicle.setColor(veh, (255,255,0))
        """
        for lane_id in traci.lane.getIDList():
            vehicles = []
            if ":" in lane_id:
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                for veh in vehicles:
                    traci.vehicle.setSpeed(veh, self.SPEED)
                continue
            #traci.polygon.setColor(lane_id,(0,0,0,0))

            if lane_id in list(self.priority.values()):
                #print(f"priorityLane:{lane_id}")
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                if not vehicles:
                    continue
                for veh in vehicles:
                    if veh == vehicles[-1]:
                        traci.vehicle.setSpeed(veh, self.SPEED)
                        traci.vehicle.setSpeedMode(veh, 7)
                        traci.vehicle.setColor(veh, (255,0,0))
                    else:
                        traci.vehicle.setSpeedMode(veh, -1)
                        traci.vehicle.setColor(veh, (255,255,0))
        """
        """
                try:
                    print(f"set polygon color {lane_id}")
                    traci.polygon.setColor(lane_id,(0,255,0,150))
                    traci.polygon.setFilled(lane_id, True)
                except traci.TraCIException as e:
                    print(e)
                """
        """
            else:
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                if not vehicles:
                    continue
                for veh in vehicles:
                    if veh == vehicles[-1]:
                        dist = traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(veh)
                        if dist <= MARGIN:
                            traci.vehicle.setSpeed(veh, 0)
                            traci.vehicle.setColor(veh, (0,255,0))
                        elif dist <= STOP_THRESHOLD:
                            # 停止線に近いがまだ余裕がある -> 徐行で詰める (例: 2m/s)
                            # これをしないと、勢いよく突っ込んできて止まりきれない
                            traci.vehicle.setSpeed(veh, 2.0) 
                            traci.vehicle.setSpeedMode(veh, 1) 
                            traci.vehicle.setColor(veh, (0, 255, 128)) # 薄い緑
                        else:
                            traci.vehicle.setSpeedMode(veh, -1)
                            traci.vehicle.setColor(veh, (255,255,0))
                    else:
                        traci.vehicle.setSpeedMode(veh, -1)
                        traci.vehicle.setColor(veh, (255,255,0))
        """

    def get_targetVeh(self, intersection):
        jx, jy = traci.junction.getPosition(intersection)
        
        in_target = {i: [] for i in self.lane_dict[intersection]["incoming"]}
        out_target = {i: [] for i in self.lane_dict[intersection]["outgoing"]}
        internal_veh = []
        for edgeID in self.lane_dict[intersection]["incoming"]:
            in_target[edgeID] = traci.edge.getLastStepVehicleIDs(edgeID)
            

        for edgeID in self.lane_dict[intersection]["outgoing"]:
            out_target[edgeID] = traci.edge.getLastStepVehicleIDs(edgeID)
            
    
        in_vs = []
        for lane in self.lane_dict[intersection]["internal"]:
            in_vs = traci.lane.getLastStepVehicleIDs(lane)
            for v in in_vs:
                internal_veh.append(v)

        return in_target, out_target, internal_veh


    def get_state(self, intersection, prev_a):
        #各車線の特徴量ベクトル
        #流入車線：3セグメント毎の車両数
        #流出車線：全体の車両数
        state = []
        in_target, out_target, internal_veh = self.get_targetVeh(intersection)
        for edge in in_target.keys():
            edge_length = traci.lane.getLength(f"{edge}_0")
            obs_len = edge_length / 3
            queue1 = 0
            queue2 = 0
            queue3 = 0
            #流入車線
            in_target_veh = in_target[edge]
            #車両密度
            for veh in in_target_veh:
                if traci.vehicle.getLanePosition(veh) < obs_len:
                    queue1 += 1
                elif traci.vehicle.getLanePosition(veh) < obs_len * 2:
                    queue2 += 1
                else: 
                    queue3 += 1
            state.extend([queue1, queue2, queue3])
                
                
        for edge in out_target.keys():
            #流出車線
            out_target_veh = out_target[edge]
            if len(out_target_veh) > 0:
                #print(len(out_target_veh))
                #車両密度
                state.append(len(out_target_veh))
                """
                vel = 0
                wait = 0
                for veh in out_target_veh:
                    vel += traci.vehicle.getSpeed(veh)
                    wait += traci.vehicle.getWaitingTime(veh)
                #平均待ち時間
                state.append(wait / len(out_target_veh))
                #平均速度
                state.append((vel / len(out_target_veh)) / self.SPEED)
                """
            #print(f"outgoing-{i}-state:{out_state[i]}")
            else:
                state.append(0)
                """
                for _ in range(3):
                    state.append(0)
                    """
        a_onehot = [0,0,0,0]
        a_onehot[prev_a] = 1
        for a in a_onehot:
            state.append(a)
        #print(f"{intersection}_state: {state}")

        return state

    def done_check(self, current_time):
        # タイムアップ
        if current_time >= self.time_limit:
            return "TimeOver"
        
        #グリッドロック検知
        for intersection in self.lane_dict.keys():
            in_target_veh, out_target_veh, internal_veh = self.get_targetVeh(intersection)
            for veh in internal_veh:
                if traci.vehicle.getWaitingTime(veh) >= 100:
                    return intersection

        # 全車両終了
        if traci.simulation.getMinExpectedNumber() == 0:
            return "Clear"
        return "Continue"

    def get_reward(self, i):
        #ペアごとにin,outのキャパシティ算出→それぞれ車両数取得後正規化→|in - out|でプレッシャー計算
        VMAX = 13
        intersection_press = 0
        pairs = self.pair_dict[i]
        for in_lane in pairs:
            in_press = traci.lane.getLastStepVehicleNumber(in_lane) / VMAX
            
            for out_lane in pairs[in_lane]:
                out_press = traci.lane.getLastStepVehicleNumber(out_lane) / VMAX
                intersection_press += abs(in_press - out_press)
    
        return intersection_press

        """
        for intersection in reward_sum.keys():
            in_target, out_target, internal_veh = self.get_targetVeh(intersection)
            #交差点内停止車両検知
            block_num = 0
            for veh in internal_veh:
                if  traci.vehicle.getSpeed(veh) < 0.1:
                    block_num += 1
            reward_sum[intersection]["block_num"] += block_num
            #待機時間計算(1台当たり)
            wait = 0
            count = 0
            for edge in in_target.keys():
                in_target_veh = in_target[edge]
                if len(in_target_veh) > 0:
                    for veh in in_target_veh:
                        wait += traci.vehicle.getWaitingTime(veh)
                        count += 1
            if count == 0:
                wait_av = 0
            else:
                wait_av = wait / count
            reward_sum[intersection]["wait_time"] += wait_av
        
        return reward_sum
        """
                

    def step(self, actions, rewards, states):
        reward_sum = {}
        done = False
        for intersection in self.intersections:
            priority_edge = self.lane_dict[intersection]["incoming"][actions[intersection]]
            priority_lane = f"{priority_edge}_0"
            self.priority[intersection] = priority_lane
        #print("pritority", self.priority)
        for i in range(self.step_num):
            self.set_speed()
            traci.simulationStep()
            #reward_sum = self.get_reward(reward_sum)
        current_time = traci.simulation.getTime()
        #print(f"reward_sum: {reward_sum}")
        #reward
        for i in self.intersections:
            rewards[i] = - self.get_reward(i)
        #done_reward
        info = self.done_check(current_time)
        
        if info == "TimeOver":
            for i in self.intersections:
                rewards[i] = -50
            done = True
        elif info in self.intersections:
            for i in self.intersections:
                if i == info:
                    rewards[i] = -100*(self.time_limit - current_time)
                else:
                    rewards[i] = -50*(self.time_limit - current_time)
            done = True
        elif info == "Clear":
            for i in self.intersections:
                rewards[i] = 50
            done = True
        
        for intersection in self.intersections:
            states[intersection] = self.get_state(intersection, actions[intersection])
        #print("rewards",rewards)
        #print("states", states)
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
            






            