import os
import time
import random
import docker
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - CHAOS_SCRIPT - %(levelname)s - %(message)s')

CHAOS_MODE = os.getenv("CHAOS_MODE")
TARGET_SERVICES = ["llm-sidecar", "orchestrator"] # Service names from docker-compose.yaml
MIN_INTERVAL_S = 90
MAX_INTERVAL_S = 180

def get_service_containers(client, service_name_pattern):
    '''
    Finds containers that match a service name pattern.
    Docker Compose often names containers like projectname_servicename_1.
    We need to find containers where the service name is part of the label or name.
    A common label is 'com.docker.compose.service'.
    '''
    containers = []
    try:
        all_containers = client.containers.list(all=True) # List all containers, including stopped ones if any
        for container in all_containers:
            # Check labels first, this is more reliable
            if container.labels.get("com.docker.compose.service") == service_name_pattern:
                containers.append(container)
                continue
            # Fallback: check name (less reliable due to potential variations)
            # Example: project_llm-sidecar_1, project-llm-sidecar-1
            # This simple check might need adjustment based on actual naming conventions
            if f"_{service_name_pattern}_" in container.name or f"-{service_name_pattern}-" in container.name:
                 if container not in containers: # Avoid duplicates if both label and name match
                    containers.append(container)
        
        if not containers:
            logging.warning(f"No containers found for service pattern: {service_name_pattern}")
        return containers
    except docker.errors.DockerException as e:
        logging.error(f"Error listing containers: {e}")
        return []

def restart_service_containers(service_name, containers_to_restart):
    if not containers_to_restart:
        logging.info(f"No active containers found for service {service_name} to restart.")
        return

    logging.info(f"CHAOS: Attempting to restart service: {service_name} (Containers: {[c.name for c in containers_to_restart]})")
    for container in containers_to_restart:
        try:
            logging.info(f"Restarting container {container.name} (ID: {container.id}) for service {service_name}...")
            container.restart(timeout=30) # Timeout for restart operation
            logging.info(f"Successfully restarted container {container.name} for service {service_name}.")
        except docker.errors.DockerException as e:
            logging.error(f"Failed to restart container {container.name} for service {service_name}: {e}")
        except Exception as e_gen:
            logging.error(f"An unexpected error occurred while restarting container {container.name}: {e_gen}")


def chaos_loop():
    if CHAOS_MODE != "1":
        logging.info("CHAOS_MODE is not '1'. Chaos script will not run.")
        return

    logging.info("CHAOS_MODE=1 detected. Starting chaos script...")
    
    try:
        client = docker.from_env()
        # Test connection
        client.ping()
        logging.info("Successfully connected to Docker daemon.")
    except docker.errors.DockerException as e:
        logging.error(f"Failed to connect to Docker daemon: {e}. Exiting chaos script.")
        return
    except Exception as e_conn_other:
        logging.error(f"An unexpected error occurred during Docker client initialization: {e_conn_other}. Exiting chaos script.")
        return


    try:
        while True:
            chosen_service_name = random.choice(TARGET_SERVICES)
            logging.info(f"CHAOS: Selected service for potential restart: {chosen_service_name}")
            
            service_containers = get_service_containers(client, chosen_service_name)

            if service_containers:
                restart_service_containers(chosen_service_name, service_containers)
            else:
                logging.warning(f"CHAOS: No containers found for service {chosen_service_name}. Skipping restart attempt.")

            sleep_duration = random.uniform(MIN_INTERVAL_S, MAX_INTERVAL_S)
            logging.info(f"CHAOS: Sleeping for {sleep_duration:.2f} seconds until next action.")
            time.sleep(sleep_duration)

    except KeyboardInterrupt:
        logging.info("CHAOS: Script interrupted by user. Exiting.")
    except Exception as e:
        logging.error(f"CHAOS: An unexpected error occurred in the main loop: {e}", exc_info=True)
    finally:
        logging.info("CHAOS: Script finished.")

if __name__ == "__main__":
    chaos_loop()
