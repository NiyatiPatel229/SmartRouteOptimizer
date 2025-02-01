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

#####################################################################
# REPLAY BUFFER
#####################################################################
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

#####################################################################
# DELIVERY STATE
#####################################################################
class DeliveryState:
    def __init__(self, shipments: List[Dict], vehicles: List[Dict], store_location: Tuple):
        self.shipments = shipments
        self.vehicles = vehicles
        self.store_location = store_location
        self.current_trips = []

    def get_nearby_shipments(self, current_location: Tuple[float, float], max_distance: float = 5.0) -> List[int]:
        """Returns indices of shipments within max_distance km of current_location"""
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

        # Encode shipments
        for i in range(MAX_SHIPMENTS):
            if i < len(self.shipments):
                s = self.shipments[i]
                lat, lon = float(s['latitude']), float(s['longitude'])
                timeslot_val = self._encode_timeslot(s['timeslot'])
                state.extend([lat, lon, timeslot_val])
            else:
                state.extend([0.0, 0.0, 0.0])

        # Encode vehicles
        for i in range(MAX_VEHICLES):
            if i < len(self.vehicles):
                v = self.vehicles[i]
                cap = 0.0
                if v['capacity'] not in [None, float('inf')]:
                    cap = v['current_capacity'] / v['capacity']
                availability = 1.0 if v['available'] else 0.0
                state.extend([availability, cap, self._vehicle_type_encoding(v['type'])])
            else:
                state.extend([0.0, 0.0, 0.0])

        return torch.FloatTensor(state)

    def _encode_timeslot(self, timeslot: str) -> float:
        start, _ = timeslot.split('-')
        h, m, _ = start.split(':')
        hour_val = int(h) + int(m) / 60.0
        return hour_val / 24.0

    def _vehicle_type_encoding(self, vtype: str) -> float:
        if vtype == '3W':
            return 3.0
        elif vtype == '4W-EV':
            return 2.0
        elif vtype == '4W':
            return 1.0
        return 0.0

#####################################################################
# NEURAL NETWORK
#####################################################################
class DeliveryNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DeliveryNetwork, self).__init__()
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

#####################################################################
# SMART ROUTE AGENT
#####################################################################
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
            masked_values = action_values + mask
            return masked_values.argmax().item()

    def get_valid_actions(self, state: DeliveryState) -> List[int]:
        valid = []
        num_shipments = len(state.shipments)
        if num_shipments == 0:
            return valid

        # Group shipments by timeslot first
        timeslot_groups = {}
        for idx, shipment in enumerate(state.shipments):
            slot = shipment['timeslot']
            if slot not in timeslot_groups:
                timeslot_groups[slot] = []
            timeslot_groups[slot].append((idx, shipment))

        for v_idx, vehicle in enumerate(state.vehicles):
            if not vehicle['available']:
                continue

            # Get vehicle's current location
            current_location = state.store_location
            current_timeslot = None
            for trip in state.current_trips:
                if trip['vehicle_idx'] == v_idx and trip['shipments']:
                    last_shipment = trip['shipments'][-1]
                    current_location = (float(last_shipment['latitude']), float(last_shipment['longitude']))
                    current_timeslot = last_shipment['timeslot']

            # For each timeslot group
            for slot, shipments in timeslot_groups.items():
                # Skip if vehicle already has shipments in different timeslot
                if current_timeslot and current_timeslot != slot:
                    continue

                # Calculate distances to all shipments in this timeslot
                distances = []
                for s_idx, shipment in shipments:
                    dist = haversine_distance(
                        current_location[0], current_location[1],
                        float(shipment['latitude']), float(shipment['longitude'])
                    )
                    # Check if within vehicle's max radius
                    if dist <= vehicle['max_radius']:
                        distances.append((s_idx, dist))

                # Sort by distance and take closest ones
                distances.sort(key=lambda x: x[1])
                for s_idx, dist in distances[:3]:  # Consider top 3 closest shipments
                    action_id = v_idx * num_shipments + s_idx
                    valid.append(action_id)

        return valid if valid else list(range(self.action_size))



    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0, self.max_reward

        experiences = self.memory.sample(batch_size)
        batch = Experience(*zip(*experiences))

        state_batch = torch.stack([s for s in batch.state]).to(self.device)
        action_batch = torch.tensor([a for a in batch.action], device=self.device)
        reward_batch = torch.tensor([r for r in batch.reward], device=self.device)
        next_state_batch = torch.stack([s for s in batch.next_state]).to(self.device)
        done_batch = torch.tensor([d for d in batch.done], device=self.device)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            next_q_values[done_batch] = 0.0
            expected_q_values = reward_batch + (self.gamma * next_q_values)

        loss = nn.SmoothL1Loss()(current_q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        max_current_reward = reward_batch.max().item()
        if max_current_reward > self.max_reward:
            self.max_reward = max_current_reward

        return loss.item(), self.max_reward

#####################################################################
# SMART ROUTE ENVIRONMENT
#####################################################################
class SmartRouteEnvironment:
    def __init__(self, shipments_data: List[Dict], vehicle_info: List[Dict], store_location: Tuple):
        self.shipments_data = shipments_data
        self.store_location = store_location
        self.vehicle_info_raw = vehicle_info
        self.agent = None
        self.reset()

    def reset(self):
        shipments = [dict(s) for s in self.shipments_data]
        vehicles = []
        for vi in self.vehicle_info_raw:
            for _ in range(vi['Number']):
                cap = vi['Shipments_Capacity']
                if cap is None:
                    cap = float('inf')
                vehicles.append({
                    'type': vi['Vehicle_Type'],
                    'capacity': cap,
                    'max_radius': vi['Max_Trip_Radius'],
                    'available': True,
                    'current_capacity': cap
                })
        self.current_state = DeliveryState(shipments, vehicles, self.store_location)
        return self.current_state.get_state_representation()

    def step(self, action: int):
        vehicle_idx = action // len(self.current_state.shipments) if self.current_state.shipments else 0
        shipment_idx = action % len(self.current_state.shipments) if self.current_state.shipments else 0

        if not self.current_state.shipments or vehicle_idx >= len(self.current_state.vehicles):
            return self.current_state.get_state_representation(), -1.0, True

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
                s = next_state.shipments[shipment_idx]
                found_trip = None
                for t in next_state.current_trips:
                    if t['vehicle_idx'] == vehicle_idx:
                        found_trip = t
                        break
                if not found_trip:
                    found_trip = {'vehicle_idx': vehicle_idx, 'shipments': []}
                    next_state.current_trips.append(found_trip)
                found_trip['shipments'].append(s)
                v['current_capacity'] -= 1
                next_state.shipments.pop(shipment_idx)
                if v['current_capacity'] <= 0:
                    v['available'] = False

        reward = self._calculate_reward(self.current_state, action, next_state)
        if not next_state.shipments or all(not veh['available'] for veh in next_state.vehicles):
            done = True

        self.current_state = next_state
        return next_state.get_state_representation(), reward, done

    def _calculate_reward(self, old_state: DeliveryState, action: int, new_state: DeliveryState) -> float:
        rew = 0.0
        if not old_state.shipments:
            return rew

        v_idx = action // len(old_state.shipments)
        s_idx = action % len(old_state.shipments)

        # Priority vehicle bonus
        if v_idx < len(old_state.vehicles):
            vt = old_state.vehicles[v_idx]['type']
            if vt == '3W':
                rew += 2.0
            elif vt == '4W-EV':
                rew += 1.0
            else:
                rew += 0.1

        # Distance-based rewards/penalties
        if v_idx < len(old_state.vehicles) and s_idx < len(old_state.shipments):
            current_location = old_state.store_location
            current_trip = None

            # Find current vehicle's trip and last location
            for trip in old_state.current_trips:
                if trip['vehicle_idx'] == v_idx:
                    current_trip = trip
                    if trip['shipments']:
                        last_shipment = trip['shipments'][-1]
                        current_location = (float(last_shipment['latitude']), float(last_shipment['longitude']))
                    break

            new_shipment = old_state.shipments[s_idx]
            distance = self._haversine_distance(
                current_location[0], current_location[1],
                float(new_shipment['latitude']), float(new_shipment['longitude'])
            )

            # Stronger penalties for distance
            if distance > 5:
                rew -= distance * 0.8  # Increased penalty
            else:
                rew += (5 - distance) * 0.4  # Increased reward for nearby shipments

            # Additional clustering reward
            if current_trip and current_trip['shipments']:
                avg_cluster_distance = 0
                cluster_points = [(float(s['latitude']), float(s['longitude']))
                                for s in current_trip['shipments']]
                cluster_points.append((float(new_shipment['latitude']), float(new_shipment['longitude'])))

                # Calculate average distance within cluster
                for i in range(len(cluster_points)):
                    for j in range(i + 1, len(cluster_points)):
                        avg_cluster_distance += self._haversine_distance(
                            cluster_points[i][0], cluster_points[i][1],
                            cluster_points[j][0], cluster_points[j][1]
                        )
                avg_cluster_distance /= len(cluster_points) if len(cluster_points) > 1 else 1

                # Reward tight clusters
                if avg_cluster_distance < 3:
                    rew += 2.0
                elif avg_cluster_distance < 5:
                    rew += 1.0

        # Capacity utilization rewards
        for i, veh in enumerate(new_state.vehicles):
            if not veh['available']:
                assigned = 0
                for t in new_state.current_trips:
                    if t['vehicle_idx'] == i:
                        assigned = len(t['shipments'])
                        break
                if veh['capacity'] != float('inf') and veh['capacity'] > 0:
                    util = assigned / float(old_state.vehicles[i]['capacity'])
                    if util >= MIN_CAPACITY_UTILIZATION:
                        rew += 4.0 * util  # Increased reward for good utilization
                    else:
                        rew -= 3.0  # Increased penalty for poor utilization

        # Time window compliance
        if not self._time_window_compliance(new_state):
            rew -= 10.0  # Increased penalty for time window violations

        return rew



    def _calculate_total_distance(self, state: DeliveryState) -> float:
        total = 0.0
        for t in state.current_trips:
            if not t['shipments']:
                continue
            lat_prev, lon_prev = state.store_location
            for s in t['shipments']:
                lat_s, lon_s = float(s['latitude']), float(s['longitude'])
                total += self._haversine_distance(lat_prev, lon_prev, lat_s, lon_s)
                lat_prev, lon_prev = lat_s, lon_s
            total += self._haversine_distance(lat_prev, lon_prev, state.store_location[0], state.store_location[1])
        return total

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = (np.sin(dlat / 2)**2) + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def _time_window_compliance(self, state: DeliveryState) -> bool:
        for t in state.current_trips:
            lat_prev, lon_prev = state.store_location
            current_time = 0
            for s in sorted(t['shipments'], key=lambda x: x['timeslot']):
                lat_s, lon_s = float(s['latitude']), float(s['longitude'])
                dist_km = self._haversine_distance(lat_prev, lon_prev, lat_s, lon_s)
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
        start_mins = int(h1) * 60 + int(m1)
        end_mins = int(h2) * 60 + int(m2)
        return start_mins, end_mins

#####################################################################
# TRAINING LOOP
#####################################################################
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

#####################################################################
# EXPORT TO EXCEL
#####################################################################
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

#####################################################################
# FOLIUM MAP DISPLAY
#####################################################################
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

#####################################################################
# QUALITY METRICS
#####################################################################
def evaluate_trip_quality(trips, store_location):
    """
    Example metrics:
    1) Total distance traveled
    2) Average capacity utilization
    3) Count of shipments delivered on time (if time window logic is used)
    """
    total_distance = 0.0
    total_capacity_util = []
    shipments_on_time = 0
    total_shipments = 0

    for t in trips:
        lat_prev, lon_prev = store_location
        for s in t['shipments']:
            lat_s, lon_s = float(s['latitude']), float(s['longitude'])
            total_distance += haversine_distance(lat_prev, lon_prev, lat_s, lon_s)
            lat_prev, lon_prev = lat_s, lon_s
        total_distance += haversine_distance(lat_prev, lon_prev, store_location[0], store_location[1])

        cap = t.get('capacity', None)
        if cap and cap != float('inf'):
            used = len(t['shipments'])
            total_capacity_util.append(used / cap)

        for s in t['shipments']:
            # Check time window compliance in a simple way
            # This is only a placeholder; if your environment calculates on-time already, you can track it
            # Here, assume shipments delivered on time if their timeslot isn't violated
            start_mins, end_mins = parse_timeslot(s['timeslot'])
            # No real calculation; just assume on-time for demonstration
            shipments_on_time += 1
            total_shipments += 1

    average_capacity_util = np.mean(total_capacity_util) if total_capacity_util else 0.0
    on_time_ratio = shipments_on_time / total_shipments if total_shipments else 0.0

    return {
        "total_distance_traveled": round(total_distance, 2),
        "average_capacity_utilization": round(average_capacity_util, 2),
        "on_time_delivery_ratio": round(on_time_ratio, 2),
    }

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def parse_timeslot(timeslot: str) -> Tuple[int, int]:
    start, end = timeslot.split('-')
    h1, m1, _ = start.split(':')
    h2, m2, _ = end.split(':')
    return (int(h1)*60 + int(m1), int(h2)*60 + int(m2))

#####################################################################
# MAIN EXECUTION (EXAMPLE)
#####################################################################
# if __name__ == "__main__":
def main1(shipments_df):
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
        num_episodes=1000
    )

    trip_df = export_to_excel(
        trained_agent=trained_agent,
        shipments_data=shipments_data,
        vehicle_info=vehicle_info,
        store_location=store_location,
        output_file='delivery_trips.xlsx'
    )

    # print("Trip Data:")
    # print(trip_df)

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
    print("Quality Metrics:", quality_metrics)

    return trip_df