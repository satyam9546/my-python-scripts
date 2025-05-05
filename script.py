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






-------------------------------exp-4-------------------------------------------

1.Implement a system where two processes communicate using the π-Calculus framework,
dynamically creating channels and exchanging messages. Ensure that the processes interact
correctly and handle concurrent execution.

    import threading
import queue
import time

# Define Process A: Sends a message and waits for a response
def process_A(channel_A, channel_B):
    print("Process A: Creating channel and sending message to Process B")
    channel_A.put("Hello from Process A")
    response = channel_B.get()  # Wait for response from Process B
    print(f"Process A: Received response: {response}")
    time.sleep(1)

# Define Process B: Waits for message, processes it, and sends a response
def process_B(channel_A, channel_B):
    message = channel_A.get()  # Wait for message from Process A
    print(f"Process B: Received message: {message}")
    print("Process B: Sending response to Process A")
    channel_B.put("Hello from Process B")
    time.sleep(1)

# Function to initialize and run the system
def main():
    # Create two channels (queues) for communication
    channel_A = queue.Queue()
    channel_B = queue.Queue()

    # Create threads for processes A and B
    thread_A = threading.Thread(target=process_A, args=(channel_A, channel_B))
    thread_B = threading.Thread(target=process_B, args=(channel_A, channel_B))

    # Start the processes (threads)
    thread_A.start()
    thread_B.start()

    # Wait for both threads to finish
    thread_A.join()
    thread_B.join()

    print("Process communication completed.")

if __name__ == "__main__":
    main()


2.Develop a Python program that models a system of three CCS processes executing actions in
parallel, ensuring synchronization where required. Introduce relabeling and restriction to study
their impact on process behavior.

import threading
import queue
import time

class CCS_Process:
    def __init__(self, name):
        self.name = name
        self.channel = queue.Queue()

    def send(self, action):
        """Simulate sending an action."""
        print(f"{self.name}: Sending action {action}")
        self.channel.put(action)

    def receive(self):
        """Simulate receiving an action."""
        action = self.channel.get()
        print(f"{self.name}: Received action {action}")
        return action

    def execute(self, action):
        """Simulate executing an action."""
        self.send(action)
        return self.receive()


def relabel(action, mapping):
    """Apply relabeling to change action names."""
    return mapping.get(action, action)


def restrict(actions, restricted_action):
    """Restrict an action from being performed."""
    return [action for action in actions if action != restricted_action]


def process_P(channel_P, channel_Q, channel_R):
    """Process P executes actions 'a' and 'b' in parallel."""
    for _ in range(2):
        print("Process P: Performing action a")
        channel_P.put('a')
        time.sleep(1)  # Simulating some delay before performing the next action
        print("Process P: Performing action b")
        channel_P.put('b')


def process_Q(channel_P, channel_Q, channel_R):
    """Process Q executes actions 'a' and 'c' in parallel."""
    for _ in range(2):
        print("Process Q: Performing action a")
        channel_Q.put('a')
        time.sleep(1)  # Simulating some delay before performing the next action
        print("Process Q: Performing action c")
        channel_Q.put('c')


def process_R(channel_P, channel_Q, channel_R):
    """Process R executes actions 'b' and 'c' in parallel."""
    for _ in range(2):
        print("Process R: Performing action b")
        channel_R.put('b')
        time.sleep(1)  # Simulating some delay before performing the next action
        print("Process R: Performing action c")
        channel_R.put('c')


def main():
    # Create channels for synchronization
    channel_P = queue.Queue()
    channel_Q = queue.Queue()
    channel_R = queue.Queue()

    # Create processes
    P = CCS_Process('P')
    Q = CCS_Process('Q')
    R = CCS_Process('R')

    # Start threads for the processes
    thread_P = threading.Thread(target=process_P, args=(channel_P, channel_Q, channel_R))
    thread_Q = threading.Thread(target=process_Q, args=(channel_P, channel_Q, channel_R))
    thread_R = threading.Thread(target=process_R, args=(channel_P, channel_Q, channel_R))

    # Start threads
    thread_P.start()
    thread_Q.start()
    thread_R.start()

    # Wait for threads to finish
    thread_P.join()
    thread_Q.join()
    thread_R.join()

    print("Process execution completed.")


if __name__ == "__main__":
    main()

3.Simulate a process algebra-based load balancer where multiple clients send requests to a central
dispatcher that distributes tasks among available workers. Verify that requests are handled
fairly without starvation.

import threading
import queue
import time

class Dispatcher:
    def __init__(self, num_workers):
        self.task_queue = queue.Queue()  # Queue for storing tasks
        self.workers = []
        self.lock = threading.Lock()  # Lock to ensure synchronization

        # Create worker threads
        for i in range(num_workers):
            worker = Worker(f"Worker-{i + 1}", self.task_queue)
            self.workers.append(worker)

    def distribute_task(self, task):
        """Distribute task to available workers."""
        self.task_queue.put(task)
        print(f"Dispatcher: Task {task} added to the queue.")

    def start_workers(self):
        """Start all worker threads."""
        for worker in self.workers:
            worker.start()

    def wait_for_completion(self):
        """Wait for all tasks to be processed by the workers."""
        for worker in self.workers:
            worker.join()


class Worker(threading.Thread):
    def __init__(self, name, task_queue):
        super().__init__()
        self.name = name
        self.task_queue = task_queue

    def run(self):
        while True:
            # Get a task from the queue (blocking until a task is available)
            task = self.task_queue.get()
            if task == "EXIT":
                break  # Worker exits when receiving the "EXIT" task
            print(f"{self.name}: Processing task {task}")
            time.sleep(2)  # Simulate task processing time
            print(f"{self.name}: Finished task {task}")
            self.task_queue.task_done()


class Client(threading.Thread):
    def __init__(self, name, dispatcher, num_tasks):
        super().__init__()
        self.name = name
        self.dispatcher = dispatcher
        self.num_tasks = num_tasks

    def run(self):
        for i in range(self.num_tasks):
            task = f"Task-{i + 1}"
            print(f"{self.name}: Sending {task} to dispatcher.")
            self.dispatcher.distribute_task(task)
            time.sleep(1)  # Simulate time between requests


def main():
    num_workers = 3  # Number of workers
    dispatcher = Dispatcher(num_workers)

    # Start the workers
    dispatcher.start_workers()

    # Create clients with varying number of tasks to simulate load balancing
    client1 = Client("Client-1", dispatcher, 5)
    client2 = Client("Client-2", dispatcher, 5)
    client3 = Client("Client-3", dispatcher, 5)

    # Start the client threads
    client1.start()
    client2.start()
    client3.start()

    # Wait for all client threads to finish
    client1.join()
    client2.join()
    client3.join()

    # Wait for all tasks to be processed by the workers
    dispatcher.task_queue.join()

    # Stop all workers by sending "EXIT" tasks
    for worker in dispatcher.workers:
        dispatcher.task_queue.put("EXIT")

    # Wait for workers to finish
    dispatcher.wait_for_completion()

    print("Load balancing completed. All tasks processed.")

if __name__ == "__main__":
    main()

4.Implement a Python-based verification system that checks whether two given finite-state
processes are equivalent using strong bisimulation. The program should take two process
descriptions as input and determine whether they exhibit the same external behavior.

class StateMachine:
    def __init__(self, name):
        self.name = name
        self.states = set()
        self.transitions = {}
        self.initial_state = None

    def add_state(self, state_name, is_initial=False):
        """Add a state to the state machine."""
        self.states.add(state_name)
        if is_initial:
            self.initial_state = state_name
        self.transitions[state_name] = []

    def add_transition(self, from_state, action, to_state):
        """Add a transition from one state to another."""
        self.transitions[from_state].append((action, to_state))

    def get_transitions(self, state):
        """Return all transitions for a given state."""
        return self.transitions[state]

    def get_initial_state(self):
        """Return the initial state."""
        return self.initial_state


def strong_bisimulation(process1, process2):
    """
    Check if two processes are strongly bisimulation equivalent.

    Arguments:
    - process1: First StateMachine
    - process2: Second StateMachine
    """
    visited = set()

    def bisimulate(state1, state2):
        if (state1, state2) in visited:
            return True
        visited.add((state1, state2))

        transitions1 = process1.get_transitions(state1)
        transitions2 = process2.get_transitions(state2)

        # Check if both processes have the same actions
        actions1 = {action for action, _ in transitions1}
        actions2 = {action for action, _ in transitions2}
        if actions1 != actions2:
            return False

        # Check if for each action, both processes can simulate each other's transitions
        for action in actions1:
            next_state1 = [to_state for a, to_state in transitions1 if a == action]
            next_state2 = [to_state for a, to_state in transitions2 if a == action]

            if not next_state1 or not next_state2:
                continue

            # Check if for each pair of next states, they are bisimulation equivalent
            for s1 in next_state1:
                for s2 in next_state2:
                    if not bisimulate(s1, s2):
                        return False

        return True

    # Start the bisimulation check from the initial states of both processes
    return bisimulate(process1.get_initial_state(), process2.get_initial_state())


def main():
    # Create the first process (Process 1)
    process1 = StateMachine("Process 1")
    process1.add_state("S0", is_initial=True)
    process1.add_state("S1")
    process1.add_state("S2")
    process1.add_transition("S0", "a", "S1")
    process1.add_transition("S1", "b", "S2")
    process1.add_transition("S2", "a", "S0")

    # Create the second process (Process 2)
    process2 = StateMachine("Process 2")
    process2.add_state("S0", is_initial=True)
    process2.add_state("S1")
    process2.add_state("S2")
    process2.add_transition("S0", "a", "S1")
    process2.add_transition("S1", "b", "S2")
    process2.add_transition("S2", "a", "S0")

    # Check bisimulation equivalence
    if strong_bisimulation(process1, process2):
        print("The processes are bisimulation equivalent.")
    else:
        print("The processes are not bisimulation equivalent.")


if __name__ == "__main__":
    main()

5.Design a producer-consumer system using CCS principles, ensuring correct message passing
and proper synchronization between the producer and the consumer while preventing
deadlocks.

    import threading
import queue
import time

# Define the shared buffer (queue)
buffer_size = 5
task_queue = queue.Queue(buffer_size)

# Producer process: Produces items and puts them in the shared queue
def producer():
    for i in range(10):
        item = f"Item-{i+1}"
        task_queue.put(item)  # Add the item to the queue
        print(f"Producer: Produced {item}")
        time.sleep(1)  # Simulate time taken to produce an item

# Consumer process: Consumes items from the shared queue
def consumer():
    while True:
        if not task_queue.empty():  # Check if there is anything to consume
            item = task_queue.get()  # Get an item from the queue
            print(f"Consumer: Consumed {item}")
            task_queue.task_done()  # Mark the task as done
            time.sleep(2)  # Simulate time taken to consume an item
        else:
            print("Consumer: Waiting for items to consume...")
            time.sleep(1)  # Wait for a while before checking the queue again

# Main function to run the producer-consumer system
def main():
    # Start the producer and consumer threads
    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    # Wait for the producer thread to finish
    producer_thread.join()

    # Wait for all items to be consumed
    task_queue.join()

    # Stop the consumer thread once all items are processed
    print("Producer finished producing items.")
    print("Consumer finished consuming items.")

if __name__ == "__main__":
    main()


--------------------------------exp-5------------------------------

1. Implement a Kripke Structure in Python and verify Computation Tree Logic (CTL)
properties.
class State:
    def __init__(self, name, propositions):
        """
        Represents a state in the Kripke structure.
        :param name: Name of the state
        :param propositions: Set of atomic propositions true in this state
        """
        self.name = name
        self.propositions = propositions  # Set of atomic propositions true in this state
        self.transitions = []  # List of states this state transitions to

    def add_transition(self, next_state):
        """Add a transition to another state."""
        self.transitions.append(next_state)

    def __str__(self):
        return f"State({self.name}, {self.propositions})"


class TransitionSystem:
    def __init__(self):
        """Represents the entire transition system (Kripke structure)."""
        self.states = {}

    def add_state(self, state):
        """Add a state to the transition system."""
        self.states[state.name] = state

    def get_state(self, state_name):
        """Get a state by its name."""
        return self.states.get(state_name)

    def __str__(self):
        return "\n".join([str(state) for state in self.states.values()])


def evaluate_property(state, property_formula):
    """
    Recursively evaluate a CTL property on the given state.
    """
    if property_formula.startswith("AG"):
        # AG p: Check if p holds in every state along every path
        prop = property_formula[2:].strip()  # Extract property after 'AG'
        if prop in state.propositions:
            # Check all transitions
            for next_state in state.transitions:
                if not evaluate_property(next_state, property_formula):
                    return False
            return True
        return False

    elif property_formula.startswith("EF"):
        # EF p: Check if p holds eventually in some state along a path
        prop = property_formula[2:].strip()  # Extract property after 'EF'
        if prop in state.propositions:
            return True
        for next_state in state.transitions:
            if evaluate_property(next_state, property_formula):
                return True
        return False

    elif property_formula.startswith("A"):
        # A(p -> F q): Check if along all paths, if p holds, eventually q will hold
        if "->" in property_formula:
            prop1, prop2 = property_formula[1:].split("->")
            prop1, prop2 = prop1.strip(), prop2.strip()
            if prop1 in state.propositions:
                return evaluate_property(state, f"EF {prop2}")
            for next_state in state.transitions:
                if not evaluate_property(next_state, f"A({prop1} -> F {prop2})"):
                    return False
            return True

    # Add more CTL operators as needed (e.g., X, F, U)
    
    return False


# Example usage
if __name__ == "__main__":
    # Create states and transitions
    s1 = State("s1", {"p"})
    s2 = State("s2", {"q"})
    s3 = State("s3", {"p", "q"})

    s1.add_transition(s2)
    s2.add_transition(s3)

    ts = TransitionSystem()
    ts.add_state(s1)
    ts.add_state(s2)
    ts.add_state(s3)

    print("Transition System:")
    print(ts)

    # Check the property AG p (Always p)
    result = evaluate_property(s1, "AG p")
    print(f"Does AG p hold? {'Yes' if result else 'No'}")

    # Check the property EF q (Eventually q)
    result = evaluate_property(s1, "EF q")
    print(f"Does EF q hold? {'Yes' if result else 'No'}")

    # Check the property A(p -> F q) (If p, eventually q)
    result = evaluate_property(s1, "A(p -> F q)")
    print(f"Does A(p -> F q) hold? {'Yes' if result else 'No'}")



2.Develop a Python-based Linear Temporal Logic (LTL) model checker for verifying safety
and liveness properties.

class State:
    def __init__(self, name, propositions):
        """
        Represents a state in the Kripke structure.
        :param name: Name of the state
        :param propositions: Set of atomic propositions true in this state
        """
        self.name = name
        self.propositions = propositions  # Set of atomic propositions true in this state
        self.transitions = []  # List of states this state transitions to

    def add_transition(self, next_state):
        """Add a transition to another state."""
        self.transitions.append(next_state)

    def __str__(self):
        return f"State({self.name}, {self.propositions})"


class TransitionSystem:
    def __init__(self):
        """Represents the entire transition system (Kripke structure)."""
        self.states = {}

    def add_state(self, state):
        """Add a state to the transition system."""
        self.states[state.name] = state

    def get_state(self, state_name):
        """Get a state by its name."""
        return self.states.get(state_name)

    def __str__(self):
        return "\n".join([str(state) for state in self.states.values()])


# LTL Model Checking Functions

def check_safety_property(state, property_formula):
    """
    Check safety property (G p) - Safety properties always hold in every state.
    LTL: G p means p must hold globally (always).
    """
    if property_formula.startswith("G"):
        # G p: check if p holds in every state along all paths
        prop = property_formula[2:].strip()  # Extract the property after 'G'
        if prop in state.propositions:
            # Check all transitions
            for next_state in state.transitions:
                if not check_safety_property(next_state, property_formula):
                    return False
            return True
        return False
    return False


def check_liveness_property(state, property_formula):
    """
    Check liveness property (F p) - Liveness properties will eventually hold.
    LTL: F p means p will eventually hold (eventually).
    """
    if property_formula.startswith("F"):
        # F p: check if p holds eventually at some point along the path
        prop = property_formula[2:].strip()  # Extract the property after 'F'
        if prop in state.propositions:
            return True
        for next_state in state.transitions:
            if check_liveness_property(next_state, property_formula):
                return True
        return False
    return False


def check_until_property(state, property_formula):
    """
    Check until property (p U q) - p holds until q becomes true.
    LTL: p U q means p holds until q becomes true.
    """
    if property_formula.startswith("U"):
        # p U q: check if p holds until q becomes true
        left, right = property_formula[2:].split("U")
        left = left.strip()
        right = right.strip()
        
        if left in state.propositions:
            if right in state.propositions:
                return True
            for next_state in state.transitions:
                if check_until_property(next_state, property_formula):
                    return True
        for next_state in state.transitions:
            if check_until_property(next_state, property_formula):
                return True
        return False
    return False


# Example usage
if __name__ == "__main__":
    # Create states and transitions
    s1 = State("s1", {"p"})
    s2 = State("s2", {"q"})
    s3 = State("s3", {"p", "q"})

    s1.add_transition(s2)
    s2.add_transition(s3)
    s3.add_transition(s1)  # Loop to ensure infinite executions

    ts = TransitionSystem()
    ts.add_state(s1)
    ts.add_state(s2)
    ts.add_state(s3)

    print("Transition System:")
    print(ts)

    # Check safety property (G p) - p is always true (globally)
    result = check_safety_property(s1, "G p")
    print(f"Does safety property (G p) hold? {'Yes' if result else 'No'}")

    # Check liveness property (F p) - p will eventually hold
    result = check_liveness_property(s1, "F p")
    print(f"Does liveness property (F p) hold? {'Yes' if result else 'No'}")

    # Check until property (p U q) - p holds until q becomes true
    result = check_until_property(s1, "p U q")
    print(f"Does until property (p U q) hold? {'Yes' if result else 'No'}")


3.Model a state transition system and check for deadlock freedom using model checking.

class StateTransitionSystem:
    def __init__(self, states, transitions):
        """
        Initialize the state transition system.
        
        :param states: A set of states
        :param transitions: A dictionary of transitions {state: [next_state, ...]}
        """
        self.states = states
        self.transitions = transitions

    def get_successors(self, state):
        """Returns the set of states reachable from the given state."""
        return self.transitions.get(state, [])

    def is_deadlock_free(self):
        """Check if the system is deadlock-free by ensuring each state has an outgoing transition."""
        for state in self.states:
            if not self.get_successors(state):
                print(f"Deadlock detected at state {state}")
                return False
        return True


def main():
    # Define the states and transitions for the system
    states = {'s0', 's1', 's2'}
    transitions = {
        's0': ['s1'],
        's1': ['s2'],
        's2': [],  # Deadlock state: no outgoing transition
    }

    # Create the state transition system
    sts = StateTransitionSystem(states, transitions)

    # Check for deadlock freedom
    if sts.is_deadlock_free():
        print("The system is deadlock-free.")
    else:
        print("The system has deadlocks.")


if __name__ == "__main__":
    main()


4.Implement a property verification tool using CTL for a given transition system.


class State:
    def __init__(self, name, propositions):
        self.name = name
        self.propositions = propositions  # Set of atomic propositions
        self.transitions = []  # List of (next_state, label) pairs

    def add_transition(self, next_state):
        self.transitions.append(next_state)

    def __str__(self):
        return f"State({self.name}, {self.propositions})"

class TransitionSystem:
    def __init__(self):
        self.states = {}

    def add_state(self, state):
        self.states[state.name] = state

    def get_state(self, state_name):
        return self.states.get(state_name)

    def __str__(self):
        return "\n".join([str(state) for state in self.states.values()])

# Define the CTL property evaluation logic

def evaluate_property(state, property_formula):
    if property_formula.startswith("AG"):
        # AG p: Check if p holds in every state along every path
        prop = property_formula[2:].strip()  # Extract property after 'AG'
        if prop in state.propositions:
            # Check all transitions
            for next_state in state.transitions:
                if not evaluate_property(next_state, property_formula):
                    return False
            return True
        return False

    elif property_formula.startswith("EF"):
        # EF p: Check if p holds eventually in some state along a path
        prop = property_formula[2:].strip()  # Extract property after 'EF'
        if prop in state.propositions:
            return True
        for next_state in state.transitions:
            if evaluate_property(next_state, property_formula):
                return True
        return False

    # Add more CTL logic for different properties as needed (e.g., A(p -> F q))

    return False


# Example usage
if __name__ == "__main__":
    # Create states and transitions
    s1 = State("s1", {"p"})
    s2 = State("s2", {"q"})
    s3 = State("s3", {"p", "q"})

    s1.add_transition(s2)
    s2.add_transition(s3)

    ts = TransitionSystem()
    ts.add_state(s1)
    ts.add_state(s2)
    ts.add_state(s3)

    # Check CTL properties
    print("Transition System:")
    print(ts)

    # Check the property AG p (Always p)
    result = evaluate_property(s1, "AG p")
    print(f"Does AG p hold? {'Yes' if result else 'No'}")

    # Check the property EF q (Eventually q)
    result = evaluate_property(s1, "EF q")
    print(f"Does EF q hold? {'Yes' if result else 'No'}")

5.Verify fairness conditions in a concurrent system using temporal logic.

class State:
    def __init__(self, name, propositions):
        """
        Represents a state in the Kripke structure.
        :param name: Name of the state
        :param propositions: Set of atomic propositions true in this state
        """
        self.name = name
        self.propositions = propositions  # Set of atomic propositions true in this state
        self.transitions = []  # List of states this state transitions to

    def add_transition(self, next_state):
        """Add a transition to another state."""
        self.transitions.append(next_state)

    def __str__(self):
        return f"State({self.name}, {self.propositions})"


class TransitionSystem:
    def __init__(self):
        """Represents the entire transition system (Kripke structure)."""
        self.states = {}

    def add_state(self, state):
        """Add a state to the transition system."""
        self.states[state.name] = state

    def get_state(self, state_name):
        """Get a state by its name."""
        return self.states.get(state_name)

    def __str__(self):
        return "\n".join([str(state) for state in self.states.values()])


def weak_fairness(state, property_formula):
    """
    Weak fairness: If a process is enabled infinitely often, it must eventually happen.
    In LTL: GF p means that p is eventually true infinitely often.
    """
    if property_formula.startswith("GF"):
        # GF p means that p must eventually hold infinitely often
        prop = property_formula[2:].strip()
        if prop in state.propositions:
            return True
        for next_state in state.transitions:
            if weak_fairness(next_state, property_formula):
                return True
        return False
    return False


def strong_fairness(state, property_formula):
    """
    Strong fairness: If p is enabled infinitely often, p must eventually happen.
    In LTL: GF p means that p must eventually happen at least once.
    """
    if property_formula.startswith("GF"):
        # GF p means that p must eventually hold infinitely often
        prop = property_formula[2:].strip()
        if prop in state.propositions:
            return True
        for next_state in state.transitions:
            if strong_fairness(next_state, property_formula):
                return True
        return False
    return False


# Example usage
if __name__ == "__main__":
    # Create states and transitions
    s1 = State("s1", {"p"})
    s2 = State("s2", {"q"})
    s3 = State("s3", {"p", "q"})

    s1.add_transition(s2)
    s2.add_transition(s3)
    s3.add_transition(s1)  # Loop to ensure infinite executions

    ts = TransitionSystem()
    ts.add_state(s1)
    ts.add_state(s2)
    ts.add_state(s3)

    print("Transition System:")
    print(ts)

    # Check weak fairness (GF p)
    result = weak_fairness(s1, "GF p")
    print(f"Does weak fairness (GF p) hold? {'Yes' if result else 'No'}")

    # Check strong fairness (GF p)
    result = strong_fairness(s1, "GF p")
    print(f"Does strong fairness (GF p) hold? {'Yes' if result else 'No'}")

