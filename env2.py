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
        
        self.roots = {"-gneE64": "gneE65",
                      "gneE17": "gneE61",
                      "gneE4": "gneE60",
                      "-gneE60": "gneE63",
                      "-gneE65":"gneE64",
                      "-gneE61": "-gneE17",
                      "-gneE62": "-gneE4",
                      "-gneE63": "gneE60"}

        self.SPEED = 5
        self.DISTANCE = self.SPEED * 10
        self.net = sumolib.net.readNet('data/intersection.net.xml')
        self.shape = []
        self.lane_dict = {}
        self.polygon_dict = []
        self.priority = {}
        self.intersections = ["gneJ41", "gneJ4", "gneJ5", "gneJ42",]
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
    
    def reset(self, is_gui=True):
        if is_gui:
            sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo-gui')
        else:
            sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo')
        sumoCmd = [sumoBinary, "-c", "data/intersection.sumocfg",  "--time-to-teleport", "-1"]
        #,"--ignore-junction-blocker", "0"
        traci.start(sumoCmd)

        for _ in range(3):
            traci.simulationStep()

    def make_vehicle(self, vehID, routeID, depart_time, vtype):
        traci.vehicle.add(vehID, routeID, depart=depart_time, typeID=vtype, departLane="best")
        traci.vehicle.setSpeed(vehID, self.SPEED)
        traci.vehicle.setMaxSpeed(vehID, self.SPEED)
    
    def make_random_route(self, step, num):
        depart = random.choice(list(self.roots.keys()))
        arrive = self.roots[depart]
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
            if ":" in lane_id:
                continue
            #traci.polygon.setColor(lane_id,(0,0,0,0))

            if lane_id in self.priority.values():
                print(f"priorityLane:{lane_id}")
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

    def step(self, actions):
        for intersection in self.intersections:
            priority_edge = self.lane_dict[intersection]["incoming"][actions[intersection]]
            priority_lane = f"{priority_edge}_0"
            self.priority[intersection] = priority_lane
        print("pritority", self.priority)
        self.set_speed()
        traci.simulationStep()
            

       
            
                





        


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
            






            