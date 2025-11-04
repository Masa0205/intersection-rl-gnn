import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import traci

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.dones[:] 

class Actor(nn.Module):
    def __init__(self, state_dim, action_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )

        self.actor.apply(self.init_weights)
    #方策に従って情報を収集するact
    def act(self, s):
        #actorNetから確率分布出力
        action_probs = self.actor(s)
        #確率操作用dist
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprobs = dist.log_prob(action)

        return action, action_logprobs
    
    def evaluate(self, s, a):
        action_probs = self.actor(s)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(a)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.critic.apply(self.init_weights)

    def forward(self, s):
        s_val = self.critic(s)
        return s_val

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

class Agent:
    def __init__(self, id, lane_info):
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 80
        self.lr_actor = 0.0003
        self.lr_critic = 0.001

        self.id = id
        self.lane_info = lane_info
        self.state_dim = len(self.lane_info["incoming"]) + len(self.lane_info["outgoing"])
        self.action_size = len(self.lane_info["incoming"])
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        print(self.device)

        self.buffer = RolloutBuffer()

        self.actor = Actor(self.state_dim, self.action_size).to(self.device)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        #クリッピングで比較するように古いネットワークを保持
        self.actor_old = Actor(self.state_dim, self.action_size).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim).to(self.device)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.MseLoss = nn.MSELoss()

    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprobs = self.actor.act(state)
            s_val = self.critic.forward(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprobs)
        self.buffer.state_values.append(s_val)
        return action.item()

    def train(self):
        # モンテカルロ法で真の報酬を計算
        rewards = []
        discounted_reward = 0
        #エピソードをはじめからたどって割引将来報酬と足して再格納
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        #報酬の正規化、計算後の報酬の平均を使って標準偏差1になるように正規化
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        #各要素をテンソル化
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # アドバンテージを求める
        advantages = rewards.detach() - old_state_values.detach()

        #エポック数分学習する
        for _ in range(self.K_epochs):
            #古い状態、行動の評価
            logprobs, dist_entropy = self.actor.evaluate(old_states, old_actions)
            state_values = self.critic.forward(old_states)

            state_values = torch.squeeze(state_values)

            #新旧方策の確率比
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            #クリップされたActor損失とCritic損失、エントロピー損失を統合して損失計算
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optim_actor.zero_grad()
            self.optim_critic.zero_grad()
            loss.mean().backward()
            self.optim_actor.step()
            self.optim_critic.step()
        #古い方策にコピー
        self.actor_old.load_state_dict(self.actor.state_dict())
        #バッファーを空にする
        self.buffer.clear()

    def get_reward(self):
        in_lanes = []
        in_veh = []
        in_num = 0
        for lane in traci.lane.getIDList():
            if ":" in lane:
                in_veh += traci.lane.getLastStepVehicleIDs(lane)
                in_num += traci.lane.getLastStepVehicleNumber(lane)

        r = - in_num * 0.1
        return r
        

    def get_state(self):
        SPEED = 5
        #各車線の特徴量ベクトル
        #流入車線：[車両密度,平均待ち時間,平均速度]
        #流出車線：[車両密度,平均待ち時間,平均速度]
        obs_len = 50
        jx, jy = traci.junction.getPosition(self.id)
        in_state = np.zeros((len(self.lane_info["incoming"]),3),dtype=np.float32)
        out_state = np.zeros((len(self.lane_info["outgoing"]),3),dtype=np.float32)
        in_vehicles = []
        in_target_veh = []
        out_vehicles = []
        out_target_veh = []
        #流入車線
        for i, edgeID in enumerate(self.lane_info["incoming"]):
            in_vehicles = traci.edge.getLastStepVehicleIDs(edgeID)
            for veh in in_vehicles:
                in_target_veh = []
                x, y = traci.vehicle.getPosition(veh)
                dis = np.sqrt((x - jx)**2 + (y - jy)**2)
                if dis <= obs_len:
                    in_target_veh.append(veh)
            if len(in_target_veh) > 0:
                #車両密度
                in_state[i, 0] = len(in_target_veh) / obs_len

                vel = 0
                wait = 0
                for veh in in_target_veh:
                    vel += traci.vehicle.getSpeed(veh)
                    wait += traci.vehicle.getWaitingTime(veh)
                #平均待ち時間
                in_state[i,1] = wait / len(in_target_veh)
                #平均速度
                in_state[i,2] = (vel / len(in_target_veh)) / SPEED
            #print(f"incoming-{i}-state:{in_state[i]}")
        
        #流出車線
        for i, edgeID in enumerate(self.lane_info["incoming"]):
            out_vehicles = traci.edge.getLastStepVehicleIDs(edgeID)
            
            for veh in out_vehicles:
                out_target_veh = []
                x, y = traci.vehicle.getPosition(veh)
                dis = np.sqrt((x - jx)**2 + (y - jy)**2)
                if dis <= obs_len:
                    out_target_veh.append(veh)
            if len(out_target_veh) > 0:
                #車両密度
                out_state[i, 0] = len(out_target_veh) / obs_len

                vel = 0
                wait = 0
                for veh in out_target_veh:
                    vel += traci.vehicle.getSpeed(veh)
                    wait += traci.vehicle.getWaitingTime(veh)
                #平均待ち時間
                out_state[i,1] = wait / len(out_target_veh)
                #平均速度
                out_state[i,2] = (vel / len(out_target_veh)) / SPEED
            #print(f"outgoing-{i}-state:{out_state[i]}")
        
        return np.vstack((in_state, out_state))

    

    def save(self, path):
        torch.save(self.actor.state_dict(), path)
        torch.save(self.critic.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.critic.load_state_dict(torch.load(path))
        