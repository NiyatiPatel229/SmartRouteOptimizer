import folium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import random
import pandas as pd
import networkx as nx
import xlsxwriter
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import math

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

TRAVEL_TIME_PER_KM = 5  # minutes
DELIVERY_TIME_PER_SHIPMENT = 10  # minutes
MIN_CAPACITY_UTILIZATION = 0.5  # 50% minimum capacity utilization

def compute_round_trip(depot: Tuple[float, float], shipments: List[Dict]) -> float:
    """
    Compute the round trip distance starting and ending at depot and visiting shipments in order.
    """
    if not shipments:
        return 0.0
    dist = 0.0
    prev = depot
    for s in shipments:
        loc = (float(s['latitude']), float(s['longitude']))
        dist += haversine_distance(prev[0], prev[1], loc[0], loc[1])
        prev = loc
    dist += haversine_distance(prev[0], prev[1], depot[0], depot[1])
    return dist 



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DeliveryState:
    def __init__(self, shipments: List[Dict], vehicles: List[Dict], store_location: Tuple):
        self.shipments = shipments
        self.vehicles = vehicles
        self.store_location = store_location
        self.current_trips = []

    def get_nearby_shipments(self, current_location: Tuple[float, float], max_distance: float = 5.0) -> List[int]:
        nearby = []
        for idx, shipment in enumerate(self.shipments):
            dist = self._haversine_distance(
                current_location[0], current_location[1],
                float(shipment['latitude']), float(shipment['longitude'])
            )
            if dist <= max_distance:
                nearby.append(idx)
        return nearby

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = (np.sin(dlat/2)**2) + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def get_state_representation(self):
        MAX_SHIPMENTS, MAX_VEHICLES = 50, 10
        state = [self.store_location[0], self.store_location[1]]

        for i in range(MAX_SHIPMENTS):
            if i < len(self.shipments):
                s = self.shipments[i]
                state.extend([float(s['latitude']), float(s['longitude']), self._encode_timeslot(s['timeslot'])])
            else:
                state.extend([0.0, 0.0, 0.0])

        for i in range(MAX_VEHICLES):
            if i < len(self.vehicles):
                v = self.vehicles[i]
                cap = v['current_capacity'] / v['capacity'] if v['capacity'] not in [None, float('inf')] else 0.0
                state.extend([1.0 if v['available'] else 0.0, cap, self._vehicle_type_encoding(v['type'])])
            else:
                state.extend([0.0, 0.0, 0.0])

        return torch.FloatTensor(state)

    def _encode_timeslot(self, timeslot: str) -> float:
        start, _ = timeslot.split('-')
        h, m, _ = start.split(':')
        return (int(h) + int(m)/60.0) / 24.0

    def _vehicle_type_encoding(self, vtype: str) -> float:
        return {'3W': 3.0, '4W-EV': 2.0, '4W': 1.0}.get(vtype, 0.0)

class DeliveryNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, action_size)
        )
        self.apply(self._init_weights)

    def forward(self, x):
        return self.network(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(0.01)

class SmartRouteAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DeliveryNetwork(state_size, action_size).to(self.device)
        self.target_net = DeliveryNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=0.0003, weight_decay=0.01)
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.max_reward = float('-inf')

    def select_action(self, state):
        valid_actions = self.get_valid_actions(state)
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            state_tensor = state.get_state_representation().to(self.device)
            action_values = self.policy_net(state_tensor)
            mask = torch.full_like(action_values, float('-inf'))
            mask[valid_actions] = 0
            return (action_values + mask).argmax().item()

    def get_valid_actions(self, state: DeliveryState) -> List[int]:
        valid = []
        num_shipments = len(state.shipments)
        if num_shipments == 0:
            return valid

        timeslot_groups = {}
        for idx, shipment in enumerate(state.shipments):
            slot = shipment['timeslot']
            timeslot_groups.setdefault(slot, []).append((idx, shipment))

        for v_idx, vehicle in enumerate(state.vehicles):
            if not vehicle['available']:
                continue

            current_location = state.store_location
            current_timeslot = None
            for trip in state.current_trips:
                if trip['vehicle_idx'] == v_idx and trip['shipments']:
                    current_location = (float(trip['shipments'][-1]['latitude']), 
                                      float(trip['shipments'][-1]['longitude']))
                    current_timeslot = trip['shipments'][-1]['timeslot']
                    break

            for slot, shipments in timeslot_groups.items():
                if current_timeslot and current_timeslot != slot:
                    continue

                distances = []
                for s_idx, shipment in shipments:
                    dist = state._haversine_distance(
                        current_location[0], current_location[1],
                        float(shipment['latitude']), float(shipment['longitude'])
                    )
                    if dist <= vehicle['max_radius']:
                        distances.append((s_idx, dist))

                distances.sort(key=lambda x: x[1])
                for s_idx, dist in distances[:3]:
                    action_id = v_idx * num_shipments + s_idx
                    valid.append(action_id)

        return valid if valid else list(range(self.action_size))

    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0, self.max_reward

        experiences = self.memory.sample(batch_size)
        batch = Experience(*zip(*experiences))

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, device=self.device)

        current_q = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0]
            next_q[done_batch] = 0.0
            expected_q = reward_batch + self.gamma * next_q

        loss = nn.SmoothL1Loss()(current_q, expected_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        max_r = reward_batch.max().item()
        self.max_reward = max(self.max_reward, max_r)
        return loss.item(), self.max_reward
    def get_valid_actions(self, state: DeliveryState) -> List[int]:
        valid = []
        num_shipments = len(state.shipments)
        if num_shipments == 0:
            return valid

        timeslot_groups = {}
        for idx, shipment in enumerate(state.shipments):
            slot = shipment['timeslot']
            timeslot_groups.setdefault(slot, []).append((idx, shipment))

        for v_idx, vehicle in enumerate(state.vehicles):
            if not vehicle['available']:
                continue

            # Get the current location for this vehicle’s trip (or depot if no trip yet)
            current_location = state.store_location
            current_trip = next((t for t in state.current_trips if t['vehicle_idx'] == v_idx), None)
            current_shipments = []
            if current_trip and current_trip['shipments']:
                last_shipment = current_trip['shipments'][-1]
                current_location = (float(last_shipment['latitude']), float(last_shipment['longitude']))
                current_shipments = current_trip['shipments']

            for slot, shipments in timeslot_groups.items():
                # If there is a current timeslot, only consider shipments in that slot.
                if current_trip and current_trip['shipments']:
                    if last_shipment['timeslot'] != slot:
                        continue

                distances = []
                for s_idx, shipment in shipments:
                    # Distance from current location to shipment
                    dist = state._haversine_distance(
                        current_location[0], current_location[1],
                        float(shipment['latitude']), float(shipment['longitude'])
                    )
                    # Estimate round trip: from current location to shipment and then back to depot.
                    est_round_trip = dist + state._haversine_distance(
                        float(shipment['latitude']), float(shipment['longitude']),
                        state.store_location[0], state.store_location[1]
                    )
                    allowed_distance = vehicle['max_radius'] * 2
                    if est_round_trip > allowed_distance:
                        # Do not consider shipments that would push the round-trip estimate beyond allowed distance.
                        continue
                    distances.append((s_idx, dist))

                # Sort by distance and choose up to 3 closest shipments.
                distances.sort(key=lambda x: x[1])
                for s_idx, _ in distances[:3]:
                    action_id = v_idx * num_shipments + s_idx
                    valid.append(action_id)

        return valid if valid else list(range(self.action_size))
class SmartRouteEnvironment:
    def __init__(self, shipments_data: List[Dict], vehicle_info: List[Dict], store_location: Tuple):
        self.shipments_data = shipments_data
        self.store_location = store_location
        self.vehicle_info_raw = vehicle_info
        self.reset()

    def reset(self):
        self.vehicles = []
        for vi in self.vehicle_info_raw:
            for _ in range(vi['Number']):
                cap = vi['Shipments_Capacity'] if vi['Shipments_Capacity'] else float('inf')
                self.vehicles.append({
                    'type': vi['Vehicle_Type'],
                    'capacity': cap,
                    'max_radius': vi['Max_Trip_Radius'],
                    'available': True,
                    'current_capacity': cap
                })
        self.current_state = DeliveryState(
            [dict(s) for s in self.shipments_data],
            self.vehicles,
            self.store_location
        )
        return self.current_state.get_state_representation()

    def step(self, action: int):
        vehicle_idx = action // len(self.current_state.shipments) if self.current_state.shipments else 0
        shipment_idx = action % len(self.current_state.shipments) if self.current_state.shipments else 0

        next_state = DeliveryState(
            self.current_state.shipments.copy(),
            [dict(v) for v in self.current_state.vehicles],
            self.current_state.store_location
        )
        next_state.current_trips = [dict(t) for t in self.current_state.current_trips]

        done = False
        reward = 0.0

        if vehicle_idx < len(next_state.vehicles) and next_state.vehicles[vehicle_idx]['available']:
            v = next_state.vehicles[vehicle_idx]
            if v['current_capacity'] >= 1:
                try:
                    s = next_state.shipments[shipment_idx]
                except IndexError:
                    return next_state.get_state_representation(), -1.0, True

                trip = next((t for t in next_state.current_trips if t['vehicle_idx'] == vehicle_idx), None)
                if not trip:
                    trip = {'vehicle_idx': vehicle_idx, 'shipments': []}
                    next_state.current_trips.append(trip)

                trip['shipments'].append(s)
                v['current_capacity'] -= 1
                del next_state.shipments[shipment_idx]
                
                if v['current_capacity'] <= 0:
                    v['available'] = False

        reward = self._calculate_reward(self.current_state, action, next_state)
        done = not next_state.shipments or all(not v['available'] for v in next_state.vehicles)
        self.current_state = next_state
        return next_state.get_state_representation(), reward, done
    def _haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    def _calculate_mst_distance(self, points):
        if not points:
            return 0.0
        G = nx.Graph()
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                dist = haversine_distance(
                    points[i][0], points[i][1],
                    points[j][0], points[j][1]
                )
                G.add_edge(i, j, weight=dist)
        mst = nx.minimum_spanning_tree(G)
        return sum(mst[u][v]['weight'] for u, v in mst.edges())

    # def _calculate_reward(self, old_state: DeliveryState, action: int, new_state: DeliveryState) -> float:
    #     rew = 0.0
    #     if not old_state.shipments:
    #         return rew

    #     v_idx = action // len(old_state.shipments)
    #     s_idx = action % len(old_state.shipments)

    #     # Vehicle type bonus
    #     if v_idx < len(old_state.vehicles):
    #         vt = old_state.vehicles[v_idx]['type']
    #         rew += {'3W': 2.0, '4W-EV': 1.0, '4W': 0.2}.get(vt, 0.0)

    #     # Distance-based rewards/penalties
    #     if v_idx < len(old_state.vehicles) and s_idx < len(old_state.shipments):
    #         current_location = old_state.store_location
    #         current_trip = next((t for t in old_state.current_trips if t['vehicle_idx'] == v_idx), None)
    #         if current_trip and current_trip['shipments']:
    #             last_shipment = current_trip['shipments'][-1]
    #             current_location = (float(last_shipment['latitude']), float(last_shipment['longitude']))

    #         new_shipment = old_state.shipments[s_idx]
    #         distance = haversine_distance(
    #             current_location[0], current_location[1],
    #             float(new_shipment['latitude']), float(new_shipment['longitude'])
    #         )

    #         # Penalize if distance exceeds vehicle's max radius
    #         if distance > old_state.vehicles[v_idx]['max_radius']:
    #             rew -= 20.0  # Severe penalty for exceeding range
    #         else:
    #             rew += (5 - distance) * 0.4  # Reward for nearby shipments

    #         # Clustering reward
    #         if current_trip and current_trip['shipments']:
    #             cluster_points = [(float(s['latitude']), float(s['longitude'])) 
    #                             for s in current_trip['shipments']]
    #             cluster_points.append((float(new_shipment['latitude']), float(new_shipment['longitude'])))
    #             avg_cluster_dist = self._calculate_mst_distance(cluster_points) / len(cluster_points)
    #             if avg_cluster_dist < 3:
    #                 rew += 2.0
    #             elif avg_cluster_dist < 5:
    #                 rew += 1.0

    #     # Capacity utilization rewards
    #     for i, veh in enumerate(new_state.vehicles):
    #         if not veh['available']:
    #             assigned = next((len(t['shipments']) for t in new_state.current_trips 
    #                            if t['vehicle_idx'] == i), 0)
    #             if veh['capacity'] != float('inf') and veh['capacity'] > 0:
    #                 util = assigned / float(old_state.vehicles[i]['capacity'])
    #                 if util >= MIN_CAPACITY_UTILIZATION:
    #                     rew += 10.0 * util  # High reward for good utilization
    #                 else:
    #                     rew -= 15.0  # Severe penalty for under-utilization

    #     # Time window compliance
    #     if not self._time_window_compliance(new_state):
    #         rew -= 10.0  # Penalty for time window violations

    #     return rew
    def _calculate_reward(self, old_state: DeliveryState, action: int, new_state: DeliveryState) -> float:
        rew = 0.0
        if not old_state.shipments:
            return rew

        # Identify vehicle and shipment indices from the action.
        v_idx = action // len(old_state.shipments)
        s_idx = action % len(old_state.shipments)

        if v_idx >= len(old_state.vehicles) or s_idx >= len(old_state.shipments):
            return -10.0  # Invalid action penalty

        vehicle = old_state.vehicles[v_idx]
        vt = vehicle['type']

        # Basic vehicle type bonus.
        rew += {'3W': 2.0, '4W-EV': 1.0, '4W': 0.2}.get(vt, 0.0)

        # Determine the current trip (if any) for this vehicle.
        current_trip = next((t for t in old_state.current_trips if t['vehicle_idx'] == v_idx), None)
        current_shipments = []
        if current_trip:
            current_shipments = current_trip['shipments']

        # The shipment being added.
        new_shipment = old_state.shipments[s_idx]

        # --- Round Trip Check ---
        depot = old_state.store_location
        round_trip_before = compute_round_trip(depot, current_shipments)
        round_trip_after = compute_round_trip(depot, current_shipments + [new_shipment])
        allowed_round_trip = vehicle['max_radius'] * 2  # Allowed round-trip distance.

        if round_trip_after > allowed_round_trip:
            # Heavy penalty if adding this shipment causes the route to exceed the allowed round-trip.
            rew -= 25.0
        else:
            # Extra bonus if vehicle type is prioritized (3W or 4W-EV) and round-trip is within limit.
            if vt in ['3W', '4W-EV']:
                rew += 10.0

        # --- Distance-Based Reward/Penalty ---
        # Compute distance from current location to new shipment.
        current_location = depot
        if current_shipments:
            last_shipment = current_shipments[-1]
            current_location = (float(last_shipment['latitude']), float(last_shipment['longitude']))
        shipment_distance = haversine_distance(
            current_location[0], current_location[1],
            float(new_shipment['latitude']), float(new_shipment['longitude'])
        )
        if shipment_distance > vehicle['max_radius']:
            rew -= 20.0  # Severe penalty for exceeding one-way range.
        else:
            rew += (5 - shipment_distance) * 0.4  # Reward for closer shipments.

        # --- Capacity Utilization Check ---
        # Estimate utilization after adding the new shipment.
        capacity = vehicle['capacity'] if vehicle['capacity'] != float('inf') else 9999
        assigned_after = len(current_shipments) + 1  # After assignment.
        utilization = assigned_after / capacity

        if utilization < MIN_CAPACITY_UTILIZATION:
            rew -= 25.0  # Heavy penalty for low capacity utilization.
        else:
            rew += 15.0 * utilization  # Bonus for good utilization.

        # --- Clustering Reward (unchanged or slightly tweaked) ---
        if current_shipments:
            cluster_points = [(float(s['latitude']), float(s['longitude'])) for s in current_shipments]
            cluster_points.append((float(new_shipment['latitude']), float(new_shipment['longitude'])))
            avg_cluster_dist = self._calculate_mst_distance(cluster_points) / len(cluster_points)
            if avg_cluster_dist < 3:
                rew += 2.0
            elif avg_cluster_dist < 5:
                rew += 1.0

        # --- Time Window Compliance ---
        if not self._time_window_compliance(new_state):
            rew -= 10.0  # Penalty for time window violations.

        return rew
    def _time_window_compliance(self, state: DeliveryState) -> bool:
        for t in state.current_trips:
            lat_prev, lon_prev = state.store_location
            current_time = 0
            for s in sorted(t['shipments'], key=lambda x: x['timeslot']):
                lat_s, lon_s = float(s['latitude']), float(s['longitude'])
                dist_km = haversine_distance(lat_prev, lon_prev, lat_s, lon_s)
                travel_time = dist_km * TRAVEL_TIME_PER_KM
                current_time += travel_time + DELIVERY_TIME_PER_SHIPMENT
                slot_start, slot_end = self._parse_timeslot(s['timeslot'])
                if current_time < slot_start:
                    current_time = slot_start
                elif current_time > slot_end:
                    return False
                lat_prev, lon_prev = lat_s, lon_s
        return True

    def _parse_timeslot(self, timeslot: str) -> Tuple[int, int]:
        start, end = timeslot.split('-')
        h1, m1, _ = start.split(':')
        h2, m2, _ = end.split(':')
        return int(h1) * 60 + int(m1), int(h2) * 60 + int(m2)

def train_smart_route_optimizer(shipments_data, vehicle_info, store_location, num_episodes=50):
    total_vehicles = sum(v['Number'] for v in vehicle_info)
    action_size = total_vehicles * len(shipments_data)

    env = SmartRouteEnvironment(shipments_data, vehicle_info, store_location)
    agent = SmartRouteAgent(state_size=2 + 3*50 + 3*10, action_size=action_size)

    reward_history = []
    best_reward = float('-inf')
    max_steps = 1000

    for e in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        steps = 0

        while True:
            if steps >= max_steps:
                break
            action = agent.select_action(env.current_state)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            steps += 1

            agent.memory.push(state, action, reward, next_state, done)
            if len(agent.memory) >= agent.batch_size:
                loss, max_r = agent.train_step(agent.batch_size)

            state = next_state
            if done:
                break

        if e % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        reward_history.append(episode_reward)

        if episode_reward > best_reward:
            best_reward = episode_reward

    return agent, reward_history

#####################################################################
# TRIP METRICS CALCULATOR
#####################################################################

class TripMetricsCalculator:
    def __init__(self, store_location):
        self.store_location = store_location
        self.TRAVEL_TIME_PER_KM = 5  # minutes
        self.DELIVERY_TIME_PER_SHIPMENT = 10  # minutes

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def _calculate_mst_dist(self, points):
        """Calculate minimum spanning tree distance for a set of points"""
        if not points:
            return 0.0

        G = nx.Graph()
        n_points = len(points)

        # Create a complete graph with distances as weights
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = self._haversine_distance(
                    points[i][0], points[i][1],
                    points[j][0], points[j][1]
                )
                G.add_edge(i, j, weight=dist)

        # Calculate minimum spanning tree
        mst = nx.minimum_spanning_tree(G)
        return sum(mst[u][v]['weight'] for u, v in mst.edges())

    def _calculate_trip_time(self, shipments, mst_distance):
        """Calculate total trip time including travel and delivery times."""
        travel_time = mst_distance * self.TRAVEL_TIME_PER_KM
        delivery_time = len(shipments) * self.DELIVERY_TIME_PER_SHIPMENT
        return travel_time + delivery_time

    def _calculate_time_utilization(self, shipments, total_trip_time):
        """Calculate time utilization as a ratio of trip time to available time window."""
        if not shipments:
            return 0.0

        # Parse time slots to get earliest start and latest end times
        time_slots = [self._parse_time_slot(s['timeslot']) for s in shipments]
        earliest_start = min(start for start, _ in time_slots)
        latest_end = max(end for _, end in time_slots)

        available_time = latest_end - earliest_start
        if available_time <= 0:
            return 0.0

        return min(1.0, total_trip_time / available_time)

    def _parse_time_slot(self, timeslot):
        """Convert timeslot string into start and end times in minutes."""
        start, end = timeslot.split('-')

        def time_to_minutes(time_str):
            h, m, _ = time_str.split(':')
            return int(h) * 60 + int(m)

        return time_to_minutes(start), time_to_minutes(end)

    def _calculate_coverage_utilization(self, mst_distance, vehicle_type, max_radius):
        """Calculate coverage utilization based on MST distance and vehicle's max radius."""
        if vehicle_type == '3W':
            max_distance = max_radius * 2  # Round trip
        elif vehicle_type == '4W-EV':
            max_distance = max_radius * 2
        else:  # Default case: '4W'
            max_distance = max_radius * 2

        return min(1.0, mst_distance / max_distance) if max_distance > 0 else 0.0

    def calculate_trip_metrics(self, trips, vehicles):
        rows = []

        for idx, trip in enumerate(trips, 1):
            vehicle_idx = trip['vehicle_idx']
            vt = vehicles[vehicle_idx]['type'] if vehicle_idx < len(vehicles) else 'Unknown'
            capacity = vehicles[vehicle_idx]['capacity'] if vehicle_idx < len(vehicles) else 1
            max_radius = vehicles[vehicle_idx]['max_radius'] if vehicle_idx < len(vehicles) else 0

            if capacity == float('inf'):
                capacity = 9999

            shipment_ids = ", ".join(str(s['id']) for s in trip['shipments'])
            latitudes = ", ".join(str(float(s['latitude'])) for s in trip['shipments'])
            longitudes = ", ".join(str(float(s['longitude'])) for s in trip['shipments'])
            time_slots = ", ".join(s['timeslot'] for s in trip['shipments'])

            shipment_locations = [(float(s['latitude']), float(s['longitude'])) for s in trip['shipments']]
            mst_distance = self._calculate_mst_dist([self.store_location] + shipment_locations) if shipment_locations else 0.0

            # Calculate trip time
            total_trip_time = self._calculate_trip_time(trip['shipments'], mst_distance)

            # Calculate time utilization (TIME_UTI)
            time_uti = self._calculate_time_utilization(trip['shipments'], total_trip_time)

            # Calculate coverage utilization (COV_UTI)
            cov_uti = self._calculate_coverage_utilization(mst_distance, vt, max_radius)

            rows.append({
                'TRIP_ID': f"TRIP_{idx}",
                'SHIPMENT_IDS': shipment_ids,
                'LATITUDES': latitudes,
                'LONGITUDES': longitudes,
                'TIME_SLOTS': time_slots,
                'SHIPMENTS': len(trip['shipments']),
                'MST_DIST': round(mst_distance, 2),
                'TRIP_TIME': round(total_trip_time, 2),
                'VEHICLE_TYPE': vt,
                'CAPACITY_UTI': round(len(trip['shipments']) / capacity, 2) if capacity else 0,
                'TIME_UTI': round(time_uti, 2),
                'COV_UTI': round(cov_uti, 2)
            })

        return pd.DataFrame(rows)

def export_to_excel(trained_agent, shipments_data, vehicle_info, store_location, output_file='delivery_trips.xlsx'):
    env = SmartRouteEnvironment(shipments_data, vehicle_info, store_location)
    env.agent = trained_agent

    state = env.reset()
    done = False
    max_steps = 1000
    steps = 0
    while not done and steps < max_steps:
        action = trained_agent.select_action(env.current_state)
        next_state, reward, done = env.step(action)
        state = next_state
        steps += 1

    trips = env.current_state.current_trips
    calc = TripMetricsCalculator(store_location)
    trip_df = calc.calculate_trip_metrics(trips, env.current_state.vehicles)

    if not trip_df.empty:
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            trip_df.to_excel(writer, index=False, sheet_name='Trips')
            worksheet = writer.sheets['Trips']

            for i, col in enumerate(trip_df.columns):
                max_len = max(trip_df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, max_len)

    return trip_df

def plot_trips_on_map(trips, store_lat, store_lon, map_filename="trips_map.html"):
    COLORS = [
        "red", "blue", "green", "purple", "orange",
        "gray", "darkred", "lightred", "darkblue",
        "lightblue", "darkgreen", "lightgreen",
        "black", "pink", "cadetblue"
    ]

    trip_map = folium.Map(location=[store_lat, store_lon], zoom_start=12)

    # Add store marker
    folium.Marker(
        location=[store_lat, store_lon],
        popup="Store/Depot",
        icon=folium.Icon(icon='home', color='blue')
    ).add_to(trip_map)

    # Sort trips by vehicle type priority
    sorted_trips = sorted(trips, key=lambda t: (
        t.get('vehicle_type', '4W'),  # Default to lowest priority
        t.get('vehicle_idx', 0)
    ), reverse=True)

    for idx, trip in enumerate(sorted_trips):
        route_color = COLORS[idx % len(COLORS)]

        # Create route including return to depot
        route_points = [(store_lat, store_lon)]

        # Sort shipments by timeslot
        sorted_shipments = sorted(trip['shipments'], key=lambda x: x['timeslot'])

        for shipment in sorted_shipments:
            lat = float(shipment['latitude'])
            lon = float(shipment['longitude'])
            route_points.append((lat, lon))

            # Add shipment marker
            folium.Marker(
                location=[lat, lon],
                popup=f"Trip {idx+1}<br>Shipment {shipment['id']}<br>Time: {shipment['timeslot']}",
                icon=folium.Icon(color=route_color, icon='info-sign')
            ).add_to(trip_map)

        # Add return to depot
        route_points.append((store_lat, store_lon))

        # Draw route line
        folium.PolyLine(
            route_points,
            color=route_color,
            weight=2,
            opacity=0.8,
            popup=f"Trip {idx+1}"
        ).add_to(trip_map)

    trip_map.save(map_filename)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (np.sin(dlat/2)**2) + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def evaluate_trip_quality(trips, store_location):
    total_distance = 0.0
    total_capacity_util = []
    shipments_on_time = 0
    total_shipments = 0

    for trip in trips:
        lat_prev, lon_prev = store_location
        current_time = 0  # Start at depot time (assumed 0)
        
        # Track delivery times for each shipment
        for shipment in trip['shipments']:
            # Calculate travel time to this shipment
            lat_s, lon_s = float(shipment['latitude']), float(shipment['longitude'])
            dist_km = haversine_distance(lat_prev, lon_prev, lat_s, lon_s)
            travel_time = dist_km * TRAVEL_TIME_PER_KM
            current_time += travel_time
            
            # Check time window compliance
            start_time, end_time = parse_timeslot(shipment['timeslot'])
            if start_time <= current_time <= end_time:
                shipments_on_time += 1
            current_time += DELIVERY_TIME_PER_SHIPMENT  # Add delivery handling time
            
            # Update previous coordinates
            lat_prev, lon_prev = lat_s, lon_s
            total_shipments += 1
        
        # Return to depot
        dist_km = haversine_distance(lat_prev, lon_prev, store_location[0], store_location[1])
        current_time += dist_km * TRAVEL_TIME_PER_KM
        total_distance += dist_km

        # Capacity utilization
        vehicle_cap = trip.get('capacity', 1)
        if vehicle_cap and vehicle_cap != float('inf'):
            used = len(trip['shipments'])
            total_capacity_util.append(used / vehicle_cap)

    average_capacity_util = np.mean(total_capacity_util) if total_capacity_util else 0.0
    on_time_ratio = shipments_on_time / total_shipments if total_shipments else 0.0

    return {
        "total_distance_traveled": round(total_distance, 2),
        "average_capacity_utilization": round(average_capacity_util, 2),
        "on_time_delivery_ratio": round(on_time_ratio, 2),
    }

def parse_timeslot(timeslot: str) -> Tuple[int, int]:
    start, end = timeslot.split('-')
    h1, m1, _ = start.split(':')
    h2, m2, _ = end.split(':')
    return (int(h1)*60 + int(m1), int(h2)*60 + int(m2))

def main1(shipments_df):  # Renamed from main1 to main
    shipments_data = [
        {
            "id": int(row["Shipment ID"]),
            "latitude": float(row["Latitude"]),
            "longitude": float(row["Longitude"]),
            "timeslot": str(row["Delivery Timeslot"])
        }
        for _, row in shipments_df.iterrows()
    ]
    vehicle_info = [
        {'Vehicle_Type': '3W', 'Number': 5, 'Shipments_Capacity': 5, 'Max_Trip_Radius': 15},
        {'Vehicle_Type': '4W-EV', 'Number': 2, 'Shipments_Capacity': 8, 'Max_Trip_Radius': 20},
        {'Vehicle_Type': '4W', 'Number': 10, 'Shipments_Capacity': 25, 'Max_Trip_Radius': 1000}
    ]
    store_location = (19.075887, 72.877911)

    trained_agent, reward_history = train_smart_route_optimizer(
        shipments_data=shipments_data,
        vehicle_info=vehicle_info,
        store_location=store_location,
        num_episodes=600
    )

    trip_df = export_to_excel(
        trained_agent=trained_agent,
        shipments_data=shipments_data,
        vehicle_info=vehicle_info,
        store_location=store_location,
        output_file='delivery_trips.xlsx'
    )

    # Folium visualization
    env = SmartRouteEnvironment(shipments_data, vehicle_info, store_location)
    env.agent = trained_agent
    env.reset()

    done = False
    max_steps = 100
    steps = 0
    while not done and steps < max_steps:
        action = trained_agent.select_action(env.current_state)
        _, _, done = env.step(action)
        steps += 1

    trips = env.current_state.current_trips
    plot_trips_on_map(trips, store_location[0], store_location[1], "trips_map.html")

    # Evaluate final trip quality
    quality_metrics = evaluate_trip_quality(trips, store_location)
    print("Trip Quality Metrics:", quality_metrics)
    return trip_df

# # Example usage
# if __name__ == "__main__":
#     # Sample DataFrame (replace with actual data loading)
#     shipments_df = pd.DataFrame({
#         "Shipment ID": [1, 2, 3],
#         "Latitude": [19.1, 19.2, 19.3],
#         "Longitude": [72.8, 72.9, 73.0],
#         "Delivery Timeslot": ["09:00:00-10:00:00", "10:00:00-11:00:00", "11:00:00-12:00:00"]
#     })
#     main(shipments_df)