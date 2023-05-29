# import time
# from kubernetes import client, config

import os, sys
base_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(base_path)


# from kube_stress_generator.job_gen import JobGenerator
# from kube_gym.utils import monitor

class SimStressGen:
    def __init__(self, scenario_file="trace2017.csv", debug=False):
        self.debug = debug

        self.scenario_file = scenario_file
        self.scenario = self.load_scenario(scenario_file)


    def load_scenario(self, scenario_file):
        # Load scenario
        scenario_path = os.path.join(base_path, "scenarios/", scenario_file)
        if self.debug:
            print("(SimStressGen) Scenario path: {}".format(scenario_path))
        scenario = []
        with open(scenario_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                scenario.append(line.strip().split(",") + [0, 0])
        if self.debug:
            print("(SimStressGen) Scenario: {}...".format(scenario[:5]))
        return scenario
    
    
    def create_pod(self, time): # time: sec-based time, scenario: list of jobs
        pod_spec_li = []
        for pod_spec in self.scenario:
            if time > int(pod_spec[1]) and pod_spec[-1] == 0:
                pod_spec[-1] = 1
                if self.debug:
                    print("(SimStressGen) Create pod: {}".format(pod_spec))
                pod_spec_li.append(pod_spec)
        return pod_spec_li
            
        
    def reset(self):
        self.scenario = self.load_scenario(self.scenario_file)
        if self.debug:
            print("(SimStressGen) Reset scenario: {}".format(self.scenario[:5]))