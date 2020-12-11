import sys
sys.path.append('./src')
sys.path.append('..')
from time import sleep
import json
from client.agent_client import AgentClient



if __name__ == "__main__":

    # set up communication
    default_agent_host = "127.0.0.1"
    default_agent_port = 2004
    default_observer_host = "127.0.0.1"
    default_observer_port = 2006
    id_num = 28888
    try:
        #Wrapper of the communicating messages
        with open('./src/client/server_client_config.json', 'r') as config:
            sc_json_config = json.load(config)
            default_agent_host = sc_json_config[0]["host"]
            default_agent_port = sc_json_config[0]["port"]
            agent_ip = default_agent_host
            agent_port = default_agent_port
    except EnvironmentError: # parent of IOError, OSError *and* WindowsError where available
        print("server_client_config.json not found")
    ar = AgentClient(agent_ip, agent_port)
    try:
            ar.connect_to_server()
    except socket.error as e:
        logger.error("Error in client-server communication: " + str(e))
    ar.configure(id_num)
    ar.ready_for_new_set()
    ar.ready_for_new_set()


    ground_truths = []
    noisy_ground_truths = []

    level = 0
    while level < 160:
        print("Loading level: " + str(level))
        ar.load_next_available_level()
        sleep(1.75)
        if level > 80:
            ground_truths.append(ar.get_ground_truth_without_screenshot()[0])
            for sample in range(100):
                noisy_ground_truths.append(ar.get_noisy_ground_truth_without_screenshot()[0])
        level += 1

    
    with open("non-novel_noisy_groundtruths_80-159_100samples_v041_aws.txt", 'w') as f:
        json.dump(noisy_ground_truths, f)

    with open("non-novel_non-noisy_groundtruths_80-159_v041_aws.txt", 'w') as f2:
        json.dump(ground_truths, f2)





