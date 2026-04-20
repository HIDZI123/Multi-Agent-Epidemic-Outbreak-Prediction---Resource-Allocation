from typing import Dict

import mesa
import numpy as np

from core.policy import PolicyManager, State


class PersonAgent(mesa.Agent):
    def __init__(self, unique_id, model, recovery_rate, hospitalized_recovery_rate):
        super().__init__(unique_id, model)
        self.state = State.SUSCEPTIBLE
        self.recovery_rate = recovery_rate
        self.hospitalized_recovery_rate = hospitalized_recovery_rate
        self.hospital_id = None

    def step(self):
        effective_transmission = self.model.base_transmission_rate * self.model.policy_manager.get_transmission_modifier()

        if self.state == State.INFECTED_UNTREATED:
            self.try_to_get_hospitalized()
            self.infect_neighbors(effective_transmission)
            self.try_to_recover(self.recovery_rate)
        elif self.state == State.HOSPITALIZED:
            self.try_to_recover(self.hospitalized_recovery_rate)

        if self.state != State.HOSPITALIZED:
            steps = self.model.grid.get_neighborhood(self.pos, moore=True)
            self.model.grid.move_agent(self, self.random.choice(steps))

    def infect_neighbors(self, transmission_rate):
        for other in self.model.grid.get_cell_list_contents([self.pos]):
            if isinstance(other, PersonAgent) and other.state == State.SUSCEPTIBLE:
                if self.random.random() < transmission_rate:
                    other.state = State.INFECTED_UNTREATED

    def try_to_recover(self, rate):
        if self.random.random() < rate:
            if self.state == State.HOSPITALIZED:
                hospital = next((h for h in self.model.hospitals if h.unique_id == self.hospital_id), None)
                if hospital:
                    hospital.discharge_patient()
                self.hospital_id = None
            self.state = State.RECOVERED

    def try_to_get_hospitalized(self):
        available_hospitals = [h for h in self.model.hospitals if not h.is_full]
        if not available_hospitals:
            return

        closest = min(available_hospitals, key=lambda h: abs(self.pos[0] - h.pos[0]) + abs(self.pos[1] - h.pos[1]))
        if self.pos == closest.pos:
            if closest.admit_patient():
                self.state = State.HOSPITALIZED
                self.hospital_id = closest.unique_id
        else:
            dx = closest.pos[0] - self.pos[0]
            dy = closest.pos[1] - self.pos[1]
            step = (self.pos[0] + int(np.sign(dx)), self.pos[1] + int(np.sign(dy)))
            step = (max(0, min(self.model.grid.width - 1, step[0])), max(0, min(self.model.grid.height - 1, step[1])))
            self.model.grid.move_agent(self, step)


class HospitalAgent(mesa.Agent):
    def __init__(self, unique_id, model, base_capacity=14):
        super().__init__(unique_id, model)
        self.base_capacity = base_capacity
        self.patients = 0

    @property
    def capacity(self):
        return int(self.base_capacity * self.model.policy_manager.get_hospital_capacity_modifier())

    @property
    def is_full(self):
        return self.patients >= self.capacity

    def admit_patient(self):
        if self.is_full:
            return False
        self.patients += 1
        return True

    def discharge_patient(self):
        if self.patients > 0:
            self.patients -= 1

    def step(self):
        return None


class EpidemicModel(mesa.Model):
    def __init__(
        self,
        N=700,
        width=35,
        height=35,
        transmission_rate=0.9,
        recovery_rate=0.04,
        hospitalized_recovery_rate=0.05,
        p_initial_infected=0.30,
        num_hospitals=3,
        hospital_capacity=14,
    ):
        super().__init__()
        self.base_transmission_rate = transmission_rate
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True
        self.hospitals = []
        self.policy_manager = PolicyManager()

        hospital_start = N + 1000
        for idx in range(num_hospitals):
            hospital = HospitalAgent(hospital_start + idx, self, hospital_capacity)
            self.hospitals.append(hospital)
            self.schedule.add(hospital)
            self.grid.place_agent(hospital, (self.random.randrange(width), self.random.randrange(height)))

        for idx in range(N):
            person = PersonAgent(idx, self, recovery_rate, hospitalized_recovery_rate)
            self.schedule.add(person)
            self.grid.place_agent(person, (self.random.randrange(width), self.random.randrange(height)))
            if self.random.random() < p_initial_infected:
                person.state = State.INFECTED_UNTREATED

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Susceptible": lambda m: self.count_states(m, PersonAgent, State.SUSCEPTIBLE),
                "Infected (Untreated)": lambda m: self.count_states(m, PersonAgent, State.INFECTED_UNTREATED),
                "Hospitalized": lambda m: self.count_states(m, PersonAgent, State.HOSPITALIZED),
                "Recovered": lambda m: self.count_states(m, PersonAgent, State.RECOVERED),
                "Total Hospital Occupancy": lambda m: sum(h.patients for h in m.hospitals),
                "Hospital Occupancy Rate": lambda m: sum(h.patients for h in m.hospitals) / max(1, sum(h.capacity for h in m.hospitals)),
                "Active Policies": lambda m: len(m.policy_manager.active_policies),
                "Effective Transmission Rate": lambda m: m.base_transmission_rate * m.policy_manager.get_transmission_modifier(),
                "Effective Hospital Capacity": lambda m: sum(h.capacity for h in m.hospitals),
            }
        )

    @staticmethod
    def count_states(model, agent_type, state):
        return sum(1 for a in model.schedule.agents if isinstance(a, agent_type) and a.state == state)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        infected = self.count_states(self, PersonAgent, State.INFECTED_UNTREATED)
        hospitalized = self.count_states(self, PersonAgent, State.HOSPITALIZED)
        if infected == 0 and hospitalized == 0:
            self.running = False

    def metrics_snapshot(self) -> Dict[str, float]:
        total_hospitalized = sum(h.patients for h in self.hospitals)
        total_capacity = sum(h.capacity for h in self.hospitals)
        infected_untreated = self.count_states(self, PersonAgent, State.INFECTED_UNTREATED)
        return {
            "hospital_occupancy_rate": total_hospitalized / max(1, total_capacity),
            "total_hospitalized": total_hospitalized,
            "total_capacity": total_capacity,
            "infected_untreated": infected_untreated,
            "effective_transmission_rate": self.base_transmission_rate * self.policy_manager.get_transmission_modifier(),
        }
