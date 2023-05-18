from kubernetes import config

# Load the Kubernetes configuration
config.load_kube_config()

if __name__ == "__main__":
    print("Load the Kubernetes configuration (Should be run on the master node)")