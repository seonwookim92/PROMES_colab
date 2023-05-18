import time
from kubernetes import client, config

import os, sys
base_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(base_path)

from kube_stress_generator.job_gen import JobGenerator

class StressGen:
    def __init__(self, silent="True", scenario_file="scenario-2023-02-27.csv"):
        self.scenario_file = scenario_file
        self.config = config.load_kube_config()
        self.batch_api = client.BatchV1Api()
        self.scenario = self.load_scenario(scenario_file)
        self.silent = True if silent == "True" else False
        if self.silent:
            debug = False

    def load_scenario(self, scenario_file):
        # Load scenario
        scenario_path = os.path.join(base_path, "scenarios/", scenario_file)
        scenario = []
        with open(scenario_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                scenario.append(line.strip().split(","))
        return scenario

    def start(self):
        scenario_start_time = time.time()

        # While running this program, checking how many minutes have passed since the start of the scenario
        idx = 0
        while idx < len(self.scenario):
            # Print the current minute from the start of the scenario
            current_elpased_time = time.time() - scenario_start_time
            current_elpased_minute = int((time.time() - scenario_start_time) / 60)
            current_elpased_second = int((time.time() - scenario_start_time) % 60)
            if not self.silent:
                print("Current elpased time: " + str(current_elpased_minute) + "m :" + str(current_elpased_second) + "s (Elapsed time: " + str(int(current_elpased_time)) + "s)")

            current_job = self.scenario[idx]
            # If the current minute is equal to the minute of the current job, create the job
            if current_elpased_time >= int(current_job[-1]):
                # Create a job
                job_generator = JobGenerator(current_job[0], current_job[1], int(current_job[2]), int(current_job[3]), config)
                job = job_generator.generate_job()
                self.batch_api.create_namespaced_job(namespace="default", body=job)
                if not self.silent:
                    print("Created a job: " + job.metadata.name)
                idx += 1

            time.sleep(5)
