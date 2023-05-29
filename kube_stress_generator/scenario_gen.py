# scenario_gen.py : Generate a scenario file to generate a sequence of stressors

import os

base_path = os.path.join(os.path.dirname(__file__), "..")

# Parameters
STRESS_LEVEL_MAX = 10 # Maximum stress level
STRESS_DURATION_MAX = 3 # minutes
STRESS_TYPE = ["cpu", "vm"] # Stress types to generate
NUM_JOBS = 1000 # Number of jobs to generate
RUN_TIME = 10 # minutes


import time, datetime
import random



# Will output the scenario file to the ./scenarios directory
# Each line should be [index, stress_type, stress_level, duration, start_time]

scenario = []

for i in range(NUM_JOBS):
    # Select a random stress type, stress level, and duration
    # stress_type = random.choice(STRESS_TYPE)
    # stress_level = random.randint(1, STRESS_LEVEL_MAX)
    # duration = random.randint(1, STRESS_DURATION_MAX)
    # start_time = random.randint(0, RUN_TIME * 60)
    start_time = random.randint(0, RUN_TIME * 60)
    duration = random.randint(1, STRESS_DURATION_MAX)
    # Randomly generate between 0.01 to STRESS_LEVEL_MAX * 0.01
    cpu = round(random.uniform(0.01, STRESS_LEVEL_MAX * 0.01), 2)
    mem = round(random.uniform(0.01, STRESS_LEVEL_MAX * 0.01), 2)

    # scenario.append([stress_type, stress_level, duration, start_time])
    scenario.append([start_time, duration, cpu, mem])

# Sort the scenario by start time (ascending) and prepend the index
scenario.sort(key=lambda x: x[0])
for i in range(len(scenario)):
    scenario[i].insert(0, i)

for line in scenario:
    print(line)

# Write the scenario to a file
filename = f"scenario-{STRESS_LEVEL_MAX}l-{STRESS_DURATION_MAX}m-{NUM_JOBS}p-{RUN_TIME}m.csv"
with open(os.path.join(base_path,f"scenarios/{filename}"), "w") as f:
    for line in scenario:
        f.write(f"{line[0]},{line[1]},{line[2]},{line[3]},{line[4]}\n")
print(f"Scenario file {filename} written to ./scenarios directory")

# Path: stress-gen/scenario_runner.py