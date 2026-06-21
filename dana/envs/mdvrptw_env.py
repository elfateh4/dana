import torch
import numpy as np
from typing import Optional, Tuple, Dict, List


class DisasterMDVRPTWEnv:
    def __init__(
        self,
        num_depots: int = 3,
        num_customers: int = 50,
        num_vehicles: int = 10,
        vehicle_capacity: float = 200.0,
        time_horizon: float = 480.0,
        disaster_prob: float = 0.05,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        self.num_depots = num_depots
        self.num_customers = num_customers
        self.num_nodes = num_depots + num_customers
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.time_horizon = time_horizon
        self.disaster_prob = disaster_prob
        self.reward_weights = reward_weights or {
            "response": 0.4,
            "satisfaction": 0.3,
            "equity": 0.2,
            "violation": -0.1,
        }
        self.reset()

    def reset(self, instance: Optional[Dict] = None) -> Dict:
        if instance is not None:
            self._load_instance(instance)
        else:
            self._generate_instance()
        self.vehicle_loc = torch.zeros(self.num_vehicles, dtype=torch.long)
        self.vehicle_time = torch.zeros(self.num_vehicles)
        self.vehicle_load = torch.zeros(self.num_vehicles)
        self.vehicle_depot = torch.arange(self.num_vehicles) % self.num_depots
        self.visited = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.visited[: self.num_depots] = True
        self.served_demand = torch.zeros(self.num_customers)
        self.current_vehicle = 0
        self.step_count = 0
        self.disaster_events = []
        self.current_distance = self.base_distance.clone()
        self.current_duration = self.base_duration.clone()
        self.done = False
        return self._get_state()

    def _load_instance(self, instance: Dict):
        self.coords = torch.as_tensor(instance["coords"], dtype=torch.float)
        self.base_distance = torch.as_tensor(
            instance["distance_matrix"], dtype=torch.float
        )
        self.base_duration = torch.as_tensor(
            instance["duration_matrix"], dtype=torch.float
        )
        self.demand = torch.as_tensor(instance["demand"], dtype=torch.float)
        self.tw_start = torch.as_tensor(instance["tw_start"], dtype=torch.float)
        self.tw_end = torch.as_tensor(instance["tw_end"], dtype=torch.float)
        self.depot_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.depot_mask[instance.get("depot_indices", list(range(self.num_depots)))] = (
            True
        )

    def _generate_instance(self):
        rng = np.random.default_rng()
        coords = rng.uniform(0, 1, size=(self.num_nodes, 2))
        dx = coords[:, None, 0] - coords[None, :, 0]
        dy = coords[:, None, 1] - coords[None, :, 1]
        dist = np.sqrt(dx**2 + dy**2).astype(np.float32)
        noise = rng.uniform(0.8, 1.2, size=(self.num_nodes, self.num_nodes)).astype(
            np.float32
        )
        np.fill_diagonal(noise, 1.0)
        self.coords = torch.tensor(coords, dtype=torch.float)
        self.base_distance = torch.tensor(dist * noise, dtype=torch.float)
        self.base_duration = self.base_distance.clone()
        self.demand = torch.randint(1, 10, (self.num_nodes,), dtype=torch.float)
        self.demand[: self.num_depots] = 0
        self.tw_start = torch.zeros(self.num_nodes)
        self.tw_end = torch.full((self.num_nodes,), self.time_horizon)
        self.depot_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.depot_mask[: self.num_depots] = True
        for i in range(self.num_depots, self.num_nodes):
            center = rng.uniform(0, self.time_horizon * 0.8)
            self.tw_start[i] = center
            self.tw_end[i] = center + rng.uniform(30, 120)

    def _get_state(self) -> Dict:
        return {
            "coords": self.coords,
            "distance_matrix": self.current_distance,
            "duration_matrix": self.current_duration,
            "demand": self.demand,
            "tw_start": self.tw_start,
            "tw_end": self.tw_end,
            "depot_mask": self.depot_mask,
            "vehicle_loc": self.vehicle_loc,
            "vehicle_time": self.vehicle_time,
            "vehicle_load": self.vehicle_load,
            "vehicle_capacity": self.vehicle_capacity,
            "vehicle_depot": self.vehicle_depot,
            "visited": self.visited,
            "served_demand": self.served_demand,
            "current_vehicle": self.current_vehicle,
            "step_count": self.step_count,
        }

    def get_mask(self) -> torch.Tensor:
        mask = self.visited.clone()
        v = self.current_vehicle
        remaining = self.vehicle_capacity - self.vehicle_load[v]
        over_capacity = self.demand > remaining
        mask = mask | over_capacity
        return mask

    def step(self, action: int) -> Tuple[Dict, float, bool]:
        if self.done:
            return self._get_state(), 0.0, True
        v = self.current_vehicle
        from_loc = self.vehicle_loc[v]
        travel_time = self.current_duration[from_loc, action]
        arrival = self.vehicle_time[v] + travel_time
        tw_penalty = 0.0
        if action >= self.num_depots:
            if arrival > self.tw_end[action]:
                tw_penalty = (arrival - self.tw_end[action]) * 0.1
            actual_arrival = max(arrival, self.tw_start[action])
        else:
            actual_arrival = arrival
        self.vehicle_loc[v] = action
        self.vehicle_time[v] = actual_arrival
        if not self.depot_mask[action]:
            self.visited[action] = True
            served = min(
                self.demand[action], self.vehicle_capacity - self.vehicle_load[v]
            )
            self.vehicle_load[v] += served
            self.served_demand[action - self.num_depots] = served / self.demand[action]
        self.step_count += 1
        self._maybe_trigger_disaster()
        all_visited = self.visited.all()
        vehicle_done = (
            self.vehicle_load[v] >= self.vehicle_capacity * 0.95
            or self.depot_mask[action]
        )
        if vehicle_done and not all_visited:
            unassigned = ~self.visited[~self.depot_mask]
            if unassigned.any():
                self.current_vehicle = (v + 1) % self.num_vehicles
                while self.current_vehicle != v:
                    nv = self.current_vehicle
                    depot_idx = self.vehicle_depot[nv]
                    self.vehicle_loc[nv] = depot_idx
                    self.vehicle_time[nv] = 0.0
                    self.vehicle_load[nv] = 0.0
                    break
        self.done = all_visited or self.step_count >= self.num_nodes * 2
        reward = self._compute_reward(tw_penalty, action)
        return self._get_state(), reward, self.done

    def _maybe_trigger_disaster(self):
        if np.random.random() < self.disaster_prob:
            event_type = np.random.choice(
                ["road_closure", "new_demand", "depot_damage"]
            )
            if event_type == "road_closure":
                i = np.random.randint(0, self.num_nodes)
                j = np.random.randint(0, self.num_nodes)
                while j == i:
                    j = np.random.randint(0, self.num_nodes)
                self.current_distance[i, j] *= 5.0
                self.current_duration[i, j] *= 5.0
                self.disaster_events.append(
                    {"type": "road_closure", "from": i, "to": j}
                )
            elif event_type == "new_demand":
                idx = np.random.randint(self.num_depots, self.num_nodes)
                if not self.visited[idx]:
                    self.demand[idx] *= 1.5
                    self.disaster_events.append({"type": "new_demand", "node": idx})
            elif event_type == "depot_damage":
                d = np.random.randint(0, self.num_depots)
                for v in range(self.num_vehicles):
                    if self.vehicle_depot[v] == d:
                        alt = (d + 1) % self.num_depots
                        self.vehicle_depot[v] = alt
                self.disaster_events.append({"type": "depot_damage", "depot": d})

    def _compute_reward(self, tw_penalty: float, action: int) -> float:
        arrived = self.vehicle_time[self.current_vehicle]
        response_reward = np.exp(-arrived / self.time_horizon)
        total_demand = self.demand[self.num_depots :].sum().item()
        served = self.served_demand.sum().item()
        satisfaction_reward = served / max(total_demand, 1)
        if self.served_demand.numel() > 1:
            equity_reward = 1.0 - torch.var(self.served_demand).item()
        else:
            equity_reward = 1.0
        violation_penalty = tw_penalty
        w = self.reward_weights
        reward = (
            w["response"] * response_reward
            + w["satisfaction"] * satisfaction_reward
            + w["equity"] * equity_reward
            + w["violation"] * violation_penalty
        )
        return reward

    def get_total_reward(self) -> float:
        total_demand = self.demand[self.num_depots :].sum().item()
        served = self.served_demand.sum().item()
        satisfaction = served / max(total_demand, 1)
        if self.served_demand.numel() > 1:
            equity = 1.0 - torch.var(self.served_demand).item()
        else:
            equity = 1.0
        avg_response = np.exp(-self.vehicle_time.mean().item() / self.time_horizon)
        w = self.reward_weights
        return (
            w["response"] * avg_response
            + w["satisfaction"] * satisfaction
            + w["equity"] * equity
        )

    def get_metrics(self) -> Dict:
        total_demand = self.demand[self.num_depots :].sum().item()
        served = self.served_demand.sum().item()
        return {
            "total_cost": self.current_distance[
                self.vehicle_loc[:-1], self.vehicle_loc[1:]
            ]
            .sum()
            .item()
            if self.vehicle_loc.shape[0] > 1
            else 0.0,
            "response_time": self.vehicle_time.mean().item(),
            "satisfaction": served / max(total_demand, 1),
            "equity": 1.0 - torch.var(self.served_demand).item()
            if self.served_demand.numel() > 1
            else 1.0,
            "num_disasters": len(self.disaster_events),
            "num_vehicles_used": (self.vehicle_load > 0).sum().item(),
        }
