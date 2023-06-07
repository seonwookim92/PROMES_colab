# scenario_gen.py : Generate a scenario file to generate a sequence of stressors

import os

base_path = os.path.join(os.path.dirname(__file__), "..")

# Parameters
STRESS_LEVEL_MAX = 5 # Maximum stress level
STRESS_DURATION_MAX = 5 # minutes
NUM_JOBS = 10000 # Number of jobs to generate
RUN_TIME = 60 # minutes


import time, datetime
import random



# Will output the scenario file to the ./scenarios directory
# Each line should be [index, stress_type, stress_level, duration, start_time]

scenario = []

for i in range(NUM_JOBS):
    start_time = random.randint(0, RUN_TIME * 60)
    duration = random.randint(1, STRESS_DURATION_MAX)
    # cpu = round(random.randint(1, STRESS_LEVEL_MAX)  * 0.1, 2)
    # mem = round(random.randint(1, STRESS_LEVEL_MAX)  * 0.1, 2)

    # Make cpu and mem resource quota unbalanced
    # If either cpu or mem has high stress level, the other will have low stress level
    random_prob = random.random()
    if random_prob < 0.4: # cpu has high stress level
        cpu = round(random.uniform(0.01, STRESS_LEVEL_MAX * 0.03), 2)
        mem = round(random.uniform(0.01, STRESS_LEVEL_MAX * 0.01), 2)
    elif random_prob < 0.8:
        cpu = round(random.uniform(0.01, STRESS_LEVEL_MAX * 0.01), 2)
        mem = round(random.uniform(0.01, STRESS_LEVEL_MAX * 0.03), 2)
    else:
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
filename = f"scenario-{STRESS_LEVEL_MAX}l-{STRESS_DURATION_MAX}m-{NUM_JOBS}p-{RUN_TIME}m_unbalanced.csv"
with open(os.path.join(base_path,f"scenarios/{filename}"), "w") as f:
    for line in scenario:
        f.write(f"{line[0]},{line[1]},{line[2]},{line[3]},{line[4]}\n")
print(f"Scenario file {filename} written to ./scenarios directory")
