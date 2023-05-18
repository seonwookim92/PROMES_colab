from kubernetes import client

class JobGenerator:
    def __init__(self, idx, stress_type, stress_level, duration, config):
        self.idx = idx
        self.stress_type = stress_type
        self.stress_level = stress_level
        self.duration = duration
        self.config = config

        if self.stress_type == "cpu":
            self.cpu_limit = f"{self.stress_level}000m"
            self.mem_limit = f"1Gi"
        elif self.stress_type == "vm":
            # self.cpu_limit = f"{max(1500, self.stress_level / 3)}m"
            self.cpu_limit = f"{self.stress_level}m"
            self.mem_limit = f"{self.stress_level + 0.5}Gi"
            self.mem_limit_stressng = f"{self.stress_level + 0.5}G"

        self.job_name = f"stressng-{self.stress_type}-{self.stress_level}l-{self.duration}m-{self.idx}th"
        self.image = "seki5405/stress-ng-image"

        if self.stress_type == "cpu":
            self.command = f"stress-ng --cpu {self.stress_level} --timeout {self.duration}m"
        elif self.stress_type == "vm":
            self.command = f"stress-ng --vm {self.stress_level} --vm-bytes {self.mem_limit_stressng} --vm-keep --timeout {self.duration}m"
        # self.command = f"stress-ng --all {self.stress_level}"

    def generate_job(self):
        job = client.V1Job(
            metadata=client.V1ObjectMeta(name=self.job_name),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name=self.job_name,
                                image=self.image,
                                command=["/bin/sh", "-c", self.command],
                                resources=client.V1ResourceRequirements(
                                    limits={"cpu": self.cpu_limit, "memory": self.mem_limit},
                                ),
                            )
                        ],
                        restart_policy="Never",
                        scheduler_name="kube-custom-scheduler",
                    )
                ),
                # active_deadline_seconds=self.duration*60,
            )
        )
        return job