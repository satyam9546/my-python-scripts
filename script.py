----------------------exp-1----------------------
1.Write a Python program to implement a simple state transition system.
class UserAccountStateMachine:
    def __init__(self):
        self.state = "Inactive"
        self.transitions = {
            "Inactive": ["Active"],
            "Active": ["Suspended", "Inactive"],
            "Suspended": ["Active"]
        }

    def get_state(self):
        return self.state

    def can_transition(self, new_state):
        return new_state in self.transitions.get(self.state, [])

    def transition_to(self, new_state):
        if self.can_transition(new_state):
            print(f"Transitioning from {self.state} to {new_state}")
            self.state = new_state
        else:
            print(f"Invalid transition from {self.state} to {new_state}")

# Example usage
if __name__ == "__main__":
    account = UserAccountStateMachine()
    
    print("Initial State:", account.get_state())
    account.transition_to("Active")
    account.transition_to("Suspended")
    account.transition_to("Inactive")  # Invalid from Suspended
    account.transition_to("Active")
    account.transition_to("Inactive")

2.Design a Python program to verify simple Boolean expressions using truth tables.
 Input a Boolean expression (e.g., (A and B) or (not A)), and generate the truth table for all
possible values of the variables.
 Compare the result against a user-provided expected truth table to verify its correctness.
import itertools

def get_variables(expr):
    # Extract variable names (assumes variables are single uppercase letters)
    return sorted(set(filter(str.isalpha, expr)))

def evaluate_expr(expr, variables, values):
    local_env = dict(zip(variables, values))
    return eval(expr, {}, local_env)

def generate_truth_table(expr):
    variables = get_variables(expr)
    truth_table = []
    for values in itertools.product([False, True], repeat=len(variables)):
        result = evaluate_expr(expr, variables, values)
        truth_table.append((values, result))
    return variables, truth_table

def print_truth_table(variables, truth_table):
    header = " | ".join(variables) + " | Result"
    print(header)
    print("-" * len(header))
    for values, result in truth_table:
        val_str = " | ".join(str(v) for v in values)
        print(f"{val_str} | {result}")

def get_user_expected_results(num_rows):
    print("\nEnter expected results for each row (True/False):")
    expected = []
    for i in range(num_rows):
        while True:
            val = input(f"Row {i+1}: ").strip().lower()
            if val in ["true", "false"]:
                expected.append(val == "true")
                break
            else:
                print("Enter 'True' or 'False'")
    return expected

def verify_results(truth_table, expected_results):
    actual_results = [result for _, result in truth_table]
    return actual_results == expected_results

def main():
    expr = input("Enter a Boolean expression (e.g., (A and B) or (not A)): ")
    variables, truth_table = generate_truth_table(expr)
    
    print("\nGenerated Truth Table:")
    print_truth_table(variables, truth_table)
    
    expected_results = get_user_expected_results(len(truth_table))
    
    if verify_results(truth_table, expected_results):
        print("\n The expression is correct as per the expected truth table.")
    else:
        print("\n The expression does NOT match the expected truth table.")

if __name__ == "__main__":
    main()


3.Implement a Python program to verify Linear Temporal Logic (LTL) formulas against a simple
finite-state machine (FSM).

    class State:
    def __init__(self, name, labels):
        self.name = name
        self.labels = set(labels)

class FSM:
    def __init__(self):
        self.states = {}
        self.transitions = {}
    
    def add_state(self, name, labels):
        self.states[name] = State(name, labels)
    
    def add_transition(self, from_state, to_state):
        self.transitions.setdefault(from_state, []).append(to_state)
    
    def get_labels(self, state_name):
        return self.states[state_name].labels

def evaluate_ltl(formula, trace, fsm):
    if formula.startswith("G "):
        sub = formula[2:]
        return all(evaluate_ltl(sub, trace[i:], fsm) for i in range(len(trace)))
    
    elif formula.startswith("F "):
        sub = formula[2:]
        return any(evaluate_ltl(sub, trace[i:], fsm) for i in range(len(trace)))
    
    elif formula.startswith("X "):
        sub = formula[2:]
        return len(trace) > 1 and evaluate_ltl(sub, trace[1:], fsm)
    
    elif " U " in formula:
        left, right = map(str.strip, formula.split(" U "))
        for i in range(len(trace)):
            if evaluate_ltl(right, trace[i:], fsm):
                return all(evaluate_ltl(left, trace[j:], fsm) for j in range(i))
        return False
    
    else:
        # Atomic proposition
        current_state = trace[0]
        return formula in fsm.get_labels(current_state)

def main():
    fsm = FSM()
    fsm.add_state("S0", ["a"])
    fsm.add_state("S1", ["b"])
    fsm.add_state("S2", ["a", "b"])
    fsm.add_transition("S0", "S1")
    fsm.add_transition("S1", "S2")
    fsm.add_transition("S2", "S0")

    # Sample trace
    trace = ["S0", "S1", "S2", "S0"]

    print("Trace:", trace)
    formula = input("Enter LTL formula (e.g., G a, F b, X b, a U b): ")

    if evaluate_ltl(formula.strip(), trace, fsm):
        print(" Formula holds over the trace.")
    else:
        print(" Formula does not hold over the trace.")

if __name__ == "__main__":
    main()
    
4.Create a Python program to simulate a reactive system for a traffic light controller with three lights:
RED, YELLOW, and GREEN.

import time

class TrafficLight:
    def __init__(self):
        self.state = "RED"  # The initial state is RED

    def change_state(self):
        """Change the traffic light state in a cyclic manner."""
        if self.state == "RED":
            self.state = "GREEN"
        elif self.state == "GREEN":
            self.state = "YELLOW"
        elif self.state == "YELLOW":
            self.state = "RED"

    def display_state(self):
        """Display the current state of the traffic light."""
        print(f"The traffic light is {self.state}")

def run_traffic_light():
    """Simulate the traffic light controller with state changes."""
    traffic_light = TrafficLight()

    while True:
        traffic_light.display_state()  # Display current light state
        if traffic_light.state == "RED":
            time.sleep(5)  # RED light for 5 seconds
        elif traffic_light.state == "GREEN":
            time.sleep(7)  # GREEN light for 7 seconds
        elif traffic_light.state == "YELLOW":
            time.sleep(2)  # YELLOW light for 2 seconds

        traffic_light.change_state()  # Change the state after the delay

if __name__ == "__main__":
    run_traffic_light()

5.Write a Python program to simulate process communication using the Communicating
Sequential Processes (CSP) model.

import threading
import time
import queue

# Sender Process: Simulates a process that sends data.
def sender(channel):
    for i in range(5):
        message = f"Message {i + 1}"
        print(f"Sender: Sending {message}")
        channel.put(message)  # Put message into the channel (send)
        time.sleep(1)  # Simulate some delay

# Receiver Process: Simulates a process that receives data.
def receiver(channel):
    for _ in range(5):
        message = channel.get()  # Get message from the channel (receive)
        print(f"Receiver: Received {message}")
        channel.task_done()  # Mark the message as processed

# Main function to run the simulation
def main():
    # Create a queue as the communication channel
    channel = queue.Queue()

    # Create threads for the sender and receiver
    sender_thread = threading.Thread(target=sender, args=(channel,))
    receiver_thread = threading.Thread(target=receiver, args=(channel,))

    # Start the threads
    sender_thread.start()
    receiver_thread.start()

    # Wait for both threads to complete
    sender_thread.join()
    receiver_thread.join()

    print("Communication finished.")

if __name__ == "__main__":
    main()




---------------------------------exp-3----------------------------
1.Write a Python program to model a client-server interaction using CCS process constructions. The
client sends a request (req) and waits for a response (res), while the server listens for req, processes it,
and responds with res. Simulate the sequential communication between both processes.


import threading
import queue
import time

def server(channel):
    while True:
        try:
            request = channel.get(timeout=10)  # Timeout added to avoid hanging indefinitely
            if request == "req":
                print("Server: Received request (req), processing...")
                time.sleep(2)
                print("Server: Responding with res")
                channel.put("res")
        except queue.Empty:
            print("Server: Timeout waiting for request")
            break

def client(channel):
    print("Client: Sending request (req) to server...")
    channel.put("req")
    try:
        response = channel.get(timeout=10)  # Timeout added for receiving response
        if response == "res":
            print("Client: Received response (res) from server")
    except queue.Empty:
        print("Client: Timeout waiting for response")

def main():
    channel = queue.Queue()
    server_thread = threading.Thread(target=server, args=(channel,))
    client_thread = threading.Thread(target=client, args=(channel,))
    server_thread.start()
    client_thread.start()

    client_thread.join()
    server_thread.join()

    print("Client-Server communication completed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Process interrupted. Exiting gracefully.")

2.Develop a Python program that defines two CCS processes, P and Q, executing actions a and b.
Apply relabeling (a → b) and restriction (\{a}) to synchronize their execution. Verify whether they
remain equivalent under strong bisimulation.

    import threading
import queue
import time

def process_P(channel):
    print("P: Performing action a")
    channel.put("a")
    time.sleep(1)

def process_Q(channel):
    print("Q: Performing action b")
    channel.put("b")
    time.sleep(1)

def relabel_action(action):
    if action == 'a':
        return 'b'
    return action

def restrict_action(actions, restricted_action):
    if restricted_action in actions:
        actions.remove(restricted_action)
    return actions

def synchronize(channel):
    P_actions = ['a']
    Q_actions = ['b']
    P_actions = restrict_action(P_actions, 'a')
    synchronized_action = relabel_action('a')

    for _ in range(3):
        P_action = channel.get()
        Q_action = channel.get()
        
        if synchronized_action == Q_action:
            print(f"Synchronization successful: {P_action} and {Q_action} are the same.")
        else:
            print(f"Synchronization failed: {P_action} and {Q_action} are different.")

def main():
    channel = queue.Queue()

    thread_P = threading.Thread(target=process_P, args=(channel,))
    thread_Q = threading.Thread(target=process_Q, args=(channel,))

    thread_P.start()
    thread_Q.start()

    synchronize(channel)

    thread_P.join()
    thread_Q.join()

if __name__ == "__main__":
    main()

3.Simulate a mobile communication system using Pi-Calculus in Python, where a parent process
dynamically spawns a child process and exchanges messages over a dynamically created channel.
Ensure the child process correctly receives and processes the messages.

    import threading
import queue
import time

def child_process(channel):
    while True:
        message = channel.get()  # Receive message from the parent process
        if message == "exit":
            break
        print(f"Child Process: Received message: {message}")
        time.sleep(1)

def parent_process():
    # Create a channel (Queue) for communication
    channel = queue.Queue()

    # Create and start the child process
    child_thread = threading.Thread(target=child_process, args=(channel,))
    child_thread.start()

    # Parent process sends messages to the child process
    for i in range(5):
        message = f"Message {i + 1}"
        print(f"Parent Process: Sending {message}")
        channel.put(message)  # Send message to the child process
        time.sleep(2)

    # Sending exit signal to child process
    channel.put("exit")

    # Wait for the child process to finish
    child_thread.join()

    print("Parent Process: Communication completed.")

if __name__ == "__main__":
    parent_process()

4.Write a Python program to define two finite-state processes in CCS and implement a bisimulation
equivalence check between them. The program should determine whether both processes exhibit the
same behavior using strong bisimulation principles from CWB.

class Process:
    def __init__(self, name, initial_state):
        self.name = name
        self.state = initial_state
        self.transitions = {}  # Dictionary to store state transitions

    def add_transition(self, state_from, action, state_to):
        """Add a transition from state_from to state_to with a given action"""
        if state_from not in self.transitions:
            self.transitions[state_from] = []
        self.transitions[state_from].append((action, state_to))

    def perform_action(self, action):
        """Perform an action and update the state"""
        if self.state in self.transitions:
            for trans_action, next_state in self.transitions[self.state]:
                if trans_action == action:
                    self.state = next_state
                    return True
        return False

    def get_possible_actions(self):
        """Get all possible actions that can be performed from the current state"""
        if self.state in self.transitions:
            return [action for action, _ in self.transitions[self.state]]
        return []

    def get_current_state(self):
        """Get the current state of the process"""
        return self.state


def strong_bisimulation(p1, p2):
    """Check if processes p1 and p2 are bisimulation equivalent"""
    visited = set()

    def bisimulate(state1, state2):
        if (state1, state2) in visited:
            return True
        visited.add((state1, state2))

        actions1 = p1.get_possible_actions()
        actions2 = p2.get_possible_actions()

        if set(actions1) != set(actions2):
            return False

        for action in actions1:
            if not p1.perform_action(action):
                continue
            if not p2.perform_action(action):
                continue
            if not bisimulate(p1.get_current_state(), p2.get_current_state()):
                return False

        return True

    return bisimulate(p1.get_current_state(), p2.get_current_state())


def main():
    # Define two processes with initial states
    process1 = Process("P1", "S0")
    process2 = Process("P2", "S0")

    # Define transitions for process1 (P1)
    process1.add_transition("S0", "a", "S1")
    process1.add_transition("S1", "b", "S2")
    process1.add_transition("S2", "a", "S0")

    # Define transitions for process2 (P2)
    process2.add_transition("S0", "a", "S1")
    process2.add_transition("S1", "b", "S2")
    process2.add_transition("S2", "a", "S0")

    # Check if both processes are bisimulation equivalent
    if strong_bisimulation(process1, process2):
        print("The processes are bisimulation equivalent.")
    else:
        print("The processes are not bisimulation equivalent.")


if __name__ == "__main__":
    main()

5.Design a Python program to simulate a fair resource scheduler for two processes (P and Q). Ensure
that both processes get access to a shared resource in a round-robin manner, preventing livelock or
starvation. Verify fairness using CCS-style modeling.

import threading
import queue
import time

# Shared resource simulation with fair round-robin scheduling
class ResourceScheduler:
    def __init__(self):
        self.channel_P = queue.Queue()
        self.channel_Q = queue.Queue()
        self.resource = "Shared Resource"  # This is the resource we're managing

    def process_P(self):
        for _ in range(3):
            self.channel_P.put("P wants resource")
            print("P: Waiting for turn to access resource...")
            self.channel_P.get()  # Wait for the signal to proceed
            print("P: Accessing shared resource")
            time.sleep(2)  # Simulate resource usage
            self.channel_Q.put("Q can now access resource")  # Notify Q it's their turn
            print("P: Released resource, giving turn to Q")

    def process_Q(self):
        for _ in range(3):
            self.channel_Q.put("Q wants resource")
            print("Q: Waiting for turn to access resource...")
            self.channel_Q.get()  # Wait for the signal to proceed
            print("Q: Accessing shared resource")
            time.sleep(2)  # Simulate resource usage
            self.channel_P.put("P can now access resource")  # Notify P it's their turn
            print("Q: Released resource, giving turn to P")

# Function to start the scheduler and processes
def main():
    scheduler = ResourceScheduler()

    # Create threads for process P and Q
    thread_P = threading.Thread(target=scheduler.process_P)
    thread_Q = threading.Thread(target=scheduler.process_Q)

    # Start both threads
    thread_P.start()
    thread_Q.start()

    # Allow the processes to start the round-robin scheduling
    scheduler.channel_P.put("P can now access resource")  # Start the process for P

    # Wait for both threads to finish
    thread_P.join()
    thread_Q.join()

    print("Fair resource scheduling completed without starvation.")

if __name__ == "__main__":
    main()
