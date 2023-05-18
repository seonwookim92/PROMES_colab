from kubernetes import client, config

import os, sys
base_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(base_path)

from kube_stress_generator.stress_gen import StressGen

silent = sys.argv[1]
scenario_file = sys.argv[2]

stress_gen = StressGen(silent=silent, scenario_file=scenario_file)
stress_gen.start()