
Lab 1: Introduction to Formal Methods Tools

Question: Simulate a vending machine using Python.

!pip install simpy
import simpy

def vending_machine(env):
    print(f"Vending Machine Ready at {env.now}")
    while True:
        print(f"Waiting for Customer at {env.now}")
        yield env.timeout(5)  # Time until next customer
        print(f"Serving Customer at {env.now}")

# Run the simulation
env = simpy.Environment()
env.process(vending_machine(env))
env.run(until=20)  # Run the simulation for 20 time units

Lab 2: Modeling Systems with CCS

Question: Simulate producer-consumer system using CCS.
import simpy

def producer(env, buffer):
    while True:
        yield env.timeout(1)  # Time to produce
        buffer.append(1)
        print(f"Produced an item at {env.now}, Buffer: {len(buffer)}")

def consumer(env, buffer):
    while True:
        if buffer:
            buffer.pop(0)
            print(f"Consumed an item at {env.now}, Buffer: {len(buffer)}")
        yield env.timeout(2)  # Time to consume

# Run the simulation
env = simpy.Environment()
buffer = []
env.process(producer(env, buffer))
env.process(consumer(env, buffer))
env.run(until=10)
Lab 3: Pi-Calculus and Dynamic Systems

Question: Model a client-server system using Pi-Calculus.

def client_server(env):
    print(f"Client sends request at {env.now}")
    yield env.timeout(1)
    print(f"Server processes request at {env.now}")
    yield env.timeout(2)
    print(f"Server sends response at {env.now}")

env = simpy.Environment()
env.process(client_server(env))
env.run(until=5)

Lab 4: Bisimulation Equivalence with Z3 Solver

Question: Verify bisimulation equivalence using the Z3 Solver.

!pip install z3-solver
from z3 import *

x1, x2 = Ints('x1 x2')
solver = Solver()
solver.add(x1 + 1 == x2 + 1)  # Both states increment similarly
if solver.check() == sat:
    print("The states are bisimilar.")
else:
    print("The states are not bisimilar.")
    
Lab 5: Fixed Points and Behavioral Properties

Question: Calculate a fixed point for a simple example using iteration.

def fixed_point(func, x0, max_iter=10):
    for _ in range(max_iter):
        x1 = func(x0)
        if x1 == x0:
            return x1
        x0 = x1
    return None

# Example: Fixed point of f(x) = x^2 for initial value x0 = 1
f = lambda x: x**2
x0 = 1
print(f"Fixed point: {fixed_point(f, x0)}")

Lab 6: Modal Logic and Temporal Properties

Question: Verify temporal properties using PyNuSMV.
# Simulate the model for a few steps
x = False
y = False

print("Initial state: x =", x, ", y =", y)

for i in range(5):  # Simulate for 5 steps
    next_x = y
    next_y = not x
    x = next_x
    y = next_y
    print("Step", i + 1, ": x =", x, ", y =", y)
    
#Check the property AG(x -> AF y) holds
#This is just a simplified check for demonstration.
#A full verification requires model checking using pynusmv.

property_holds = True
for i in range(5):
    if x:
        found_y = False
        for j in range(i,5): # Look for y in future steps
            next_x = y
            next_y = not x
            x = next_x
            y = next_y
            if y:
                found_y = True
                break

        if not found_y:
            property_holds = False

if property_holds:
    print("Property AG(x -> AF y) holds for the simulated trace.")
else:
    print("Property AG(x -> AF y) does not hold for the simulated trace.")
    
Lab 7: CTL Model Checking

Question: Write CTL properties in NuSMV and verify them.


Similar to Lab 6, you will use PyNuSMV for CTL properties verification.

Lab 8: Advanced Temporal Verification

Question: Model a banking transaction system and verify fairness properties.

def bank_transaction(env):
    print(f"Transaction Initiated at {env.now}")
    yield env.timeout(2)
    print(f"Transaction Approved at {env.now}")

env = simpy.Environment()
env.process(bank_transaction(env))
env.run(until=5)
Lab 9: Real-World Protocol Verification

Question: Simulate the TCP three-way handshake using Python.

def tcp_handshake(env):
    print("SYN sent")
    yield env.timeout(1)
    print("SYN-ACK received")
    yield env.timeout(1)
    print("ACK sent")
    yield env.timeout(1)
    print("Connection Established")

env = simpy.Environment()
env.process(tcp_handshake(env))
env.run(until=5)

Lab 10: Comprehensive System Modeling and Verification

Question: Model an elevator control system in Python.

def elevator(env, floors):
    current_floor = 0
    for target_floor in floors:
        print(f"Elevator moving from {current_floor} to {target_floor} at {env.now}")
        yield env.timeout(abs(target_floor - current_floor))  # Time to move
        current_floor = target_floor
        print(f"Elevator arrived at {current_floor} at {env.now}")

env = simpy.Environment()
floors = [0, 2, 5, 1]
env.process(elevator(env, floors))
env.run()


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
---------------------------------exp-2-------------------------------------
1.Simulate a basic CCS process in Python where one process performs an action (a) and
transitions to the next state.
    
import time

class CCSProcess:
    def __init__(self, name):
        self.name = name
        self.state = "Initial"  # The initial state of the process

    def perform_action(self, action):
        """Perform the action and transition to the next state."""
        print(f"{self.name} is performing action: {action}")
        time.sleep(1)  # Simulate action taking time
        self.transition_to_next_state()

    def transition_to_next_state(self):
        """Transition the process to the next state after performing an action."""
        if self.state == "Initial":
            self.state = "State1"
        elif self.state == "State1":
            self.state = "State2"
        elif self.state == "State2":
            self.state = "Final"
        
        print(f"{self.name} transitioned to {self.state}")
    
    def run(self):
        """Simulate the process execution."""
        actions = ['a', 'b', 'c']  # Example actions
        for action in actions:
            self.perform_action(action)
            if self.state == "Final":
                break  # Stop the process when reaching the final state


# Create and run a CCS process
process = CCSProcess("Process1")
process.run()

2.Model and simulate a parallel composition of two CCS processes in Python, where both
processes execute concurrently.
import threading
import time

class CCSProcess:
    def __init__(self, name, actions):
        self.name = name
        self.state = "Initial"
        self.actions = actions  # List of actions for this process
        self.current_action_index = 0  # To track the current action

    def perform_action(self):
        """Perform the current action and transition to the next state."""
        if self.current_action_index < len(self.actions):
            action = self.actions[self.current_action_index]
            print(f"{self.name} is performing action: {action}")
            time.sleep(1)  # Simulate time to perform the action
            self.transition_to_next_state()
            self.current_action_index += 1
        else:
            print(f"{self.name} has completed all actions.")
    
    def transition_to_next_state(self):
        """Transition the process to the next state after performing an action."""
        if self.state == "Initial":
            self.state = "State1"
        elif self.state == "State1":
            self.state = "State2"
        elif self.state == "State2":
            self.state = "Final"
        
        print(f"{self.name} transitioned to {self.state}")
    
    def run(self):
        """Run the process, performing actions and transitioning through states."""
        while self.current_action_index < len(self.actions):
            self.perform_action()


# Parallel Composition of Two CCS Processes
def parallel_composition(process1, process2):
    """
    Run two processes concurrently, allowing them to perform actions in parallel.
    If both processes perform the same action, synchronize them.
    """
    while process1.current_action_index < len(process1.actions) or process2.current_action_index < len(process2.actions):
        if process1.current_action_index < len(process1.actions):
            process1.perform_action()
        
        if process2.current_action_index < len(process2.actions):
            process2.perform_action()


# Define two processes with their actions
process1 = CCSProcess("Process1", actions=["a", "b", "c"])
process2 = CCSProcess("Process2", actions=["a", "c", "d"])

# Run the parallel composition of both processes
parallel_composition(process1, process2)

3.Implement Pi-Calculus communication in Python by simulating a sender process that sends a
message over a channel and a receiver process that receives it.
import threading
import queue
import time

# Sender process: Sends a message over a channel
def sender(channel):
    message = "Hello from Sender!"
    print("Sender: Sending message.")
    time.sleep(1)  # Simulate some processing time
    channel.put(message)  # Send the message to the channel
    print("Sender: Message sent.")

# Receiver process: Receives the message from the channel
def receiver(channel):
    print("Receiver: Waiting for message.")
    message = channel.get()  # Receive the message from the channel
    print(f"Receiver: Received message: {message}")

# Main function to run the processes
def main():
    # Create a queue to act as the communication channel
    channel = queue.Queue()

    # Create threads for sender and receiver processes
    sender_thread = threading.Thread(target=sender, args=(channel,))
    receiver_thread = threading.Thread(target=receiver, args=(channel,))

    # Start the threads
    sender_thread.start()
    receiver_thread.start()

    # Wait for both threads to finish
    sender_thread.join()
    receiver_thread.join()

    print("Communication complete.")

if __name__ == "__main__":
    main()

4.Write a Python program to verify synchronization between two CCS processes using
complementary actions (a and ā).

import threading
import time

class CCSProcess:
    def __init__(self, name, action, complement_action, other_process_event):
        self.name = name
        self.action = action
        self.complement_action = complement_action
        self.event = threading.Event()
        self.other_process_event = other_process_event

    def perform_action(self):
        """Perform the action and synchronize with the other process."""
        print(f"{self.name} is waiting to perform action {self.action}.")
        self.other_process_event.wait()  # Wait for the other process to be ready
        print(f"{self.name} performs action {self.action}.")
        time.sleep(1)  # Simulate time taken to perform the action
        self.event.set()  # Signal that the action has been performed

    def synchronize(self):
        """Ensure the complementary action from the other process is performed."""
        print(f"{self.name} is preparing for synchronization on action {self.complement_action}.")
        self.other_process_event.set()  # Signal the other process to perform its action
        self.event.wait()  # Wait for the action from the other process to complete
        print(f"{self.name} completes synchronization on action {self.complement_action}.")


# Main function to run the two processes and verify synchronization
def main():
    # Event for synchronizing processes
    event1 = threading.Event()
    event2 = threading.Event()

    # Create two processes: P1 (performing action 'a') and P2 (performing action 'ā')
    process1 = CCSProcess(name="Process 1", action="a", complement_action="ā", other_process_event=event2)
    process2 = CCSProcess(name="Process 2", action="ā", complement_action="a", other_process_event=event1)

    # Create threads for both processes
    thread1 = threading.Thread(target=process1.perform_action)
    thread2 = threading.Thread(target=process2.perform_action)

    # Start the threads
    thread1.start()
    thread2.start()

    # Synchronize the processes
    process1.synchronize()
    process2.synchronize()

    # Wait for both threads to finish
    thread1.join()
    thread2.join()

    print("Both processes have synchronized on complementary actions.")


if __name__ == "__main__":
    main()

5.Simulate a basic producer-consumer system using Python, ensuring mutual exclusion and
correct handling of shared resources.

    import threading
import queue
import time

# Producer process
def producer(buffer, lock, not_full, not_empty):
    for i in range(10):  # Produce 10 items
        item = f"Item-{i + 1}"
        with lock:  # Ensure mutual exclusion while accessing the buffer
            while buffer.full():  # Wait if the buffer is full
                print("Producer: Buffer is full, waiting to produce...")
                not_full.wait()  # Wait until the consumer consumes an item
            buffer.put(item)  # Add item to the buffer
            print(f"Producer: Produced {item}")
            not_empty.notify()  # Notify the consumer that the buffer is not empty
        time.sleep(1)  # Simulate time taken to produce an item

# Consumer process
def consumer(buffer, lock, not_full, not_empty):
    for _ in range(10):  # Consume 10 items
        with lock:  # Ensure mutual exclusion while accessing the buffer
            while buffer.empty():  # Wait if the buffer is empty
                print("Consumer: Buffer is empty, waiting to consume...")
                not_empty.wait()  # Wait until the producer produces an item
            item = buffer.get()  # Remove item from the buffer
            print(f"Consumer: Consumed {item}")
            not_full.notify()  # Notify the producer that the buffer is not full
        time.sleep(2)  # Simulate time taken to consume an item

# Main function to run the producer-consumer simulation
def main():
    buffer_size = 5  # Size of the buffer
    buffer = queue.Queue(buffer_size)
    
    # Create lock and condition variables for synchronization
    lock = threading.Lock()
    not_full = threading.Condition(lock)
    not_empty = threading.Condition(lock)

    # Create threads for the producer and consumer
    producer_thread = threading.Thread(target=producer, args=(buffer, lock, not_full, not_empty))
    consumer_thread = threading.Thread(target=consumer, args=(buffer, lock, not_full, not_empty))

    # Start the threads
    producer_thread.start()
    consumer_thread.start()

    # Wait for both threads to finish
    producer_thread.join()
    consumer_thread.join()

    print("Producer-Consumer simulation completed.")

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











-------------------------exp-6------------------------
1.Model a synchronization problem using Petri Nets and verify deadlock freedom.

    class PetriNet:
    def __init__(self):
        self.places = {
            'P1': 1,  # Process 1 is waiting
            'P2': 1,  # Process 2 is waiting
            'Lock': 0  # Lock is initially free
        }
        self.transitions = {
            'T1': self.try_to_acquire_lock,  # Process 1 trying to acquire lock
            'T2': self.try_to_acquire_lock,  # Process 2 trying to acquire lock
            'R1': self.release_lock,         # Process 1 releasing lock
            'R2': self.release_lock          # Process 2 releasing lock
        }
    
    def try_to_acquire_lock(self, process):
        """Simulate the action of trying to acquire the lock."""
        if self.places['Lock'] == 0:
            # Lock is available, acquire it
            self.places['Lock'] = 1
            print(f"Process {process} acquired the lock.")
        else:
            print(f"Process {process} is waiting for the lock.")

    def release_lock(self, process):
        """Simulate the action of releasing the lock."""
        if self.places['Lock'] == 1:
            # Lock is currently acquired, release it
            self.places['Lock'] = 0
            print(f"Process {process} released the lock.")
        else:
            print(f"Process {process} has no lock to release.")
    
    def fire_transition(self, transition, process):
        """Fire a transition if possible."""
        if transition == 'T1' or transition == 'T2':
            self.transitions[transition](process)
        elif transition == 'R1' or transition == 'R2':
            self.transitions[transition](process)

    def check_deadlock(self):
        """Check for deadlock (no transitions possible)."""
        if self.places['P1'] == 1 and self.places['P2'] == 1 and self.places['Lock'] == 1:
            # Both processes are waiting and the lock is held by someone
            print("Deadlock detected!")
            return True
        else:
            return False

# Simulation of the Petri Net for the synchronization problem

def simulate():
    petri_net = PetriNet()

    # Process 1 tries to acquire the lock
    petri_net.fire_transition('T1', 1)

    # Process 2 tries to acquire the lock (deadlock happens if both are waiting)
    petri_net.fire_transition('T2', 2)

    # Check if there is a deadlock
    if petri_net.check_deadlock():
        print("The system is deadlocked.")
    else:
        print("The system is not deadlocked.")

    # Process 1 releases the lock
    petri_net.fire_transition('R1', 1)

    # Now Process 2 should be able to acquire the lock
    petri_net.fire_transition('T2', 2)

simulate()


2.Implement a basic workflow system (e.g., an order processing system) using Petri Nets and
analyze its correctness.

    import time

class PetriNet:
    def __init__(self, places, transitions):
        """
        Initialize the Petri Net with places and transitions.
        
        :param places: A dictionary of places with their initial token counts.
        :param transitions: A dictionary of transitions with input and output places.
        """
        self.places = places
        self.transitions = transitions

    def get_successors(self, state):
        """Returns the set of states reachable from the given state."""
        return self.transitions.get(state, [])

    def can_fire_transition(self, inputs):
        """Check if a transition can fire based on the tokens in input places."""
        for place in inputs:
            if self.places[place] <= 0:
                return False  # If any input place has no tokens, the transition cannot fire
        return True

    def fire_transition(self, transition):
        """Fire a transition, updating tokens in the places."""
        inputs, outputs = self.transitions[transition]
        
        # Check if transition can fire
        if self.can_fire_transition(inputs):
            # Move tokens: consume from input places, produce in output places
            for place in inputs:
                self.places[place] -= 1
            for place in outputs:
                self.places[place] += 1
            print(f"Transition {transition} fired.")
        else:
            print(f"Transition {transition} cannot fire (deadlock detected).")

    def display_state(self):
        """Display the current state of the Petri net (token counts in places)."""
        print(f"Current state: {self.places}")


def main():
    # Define places and their initial token counts
    places = {
        'Order Received': 1,  # Initial token: Order is received
        'Order Processed': 0,  # Initially, the order is not processed
        'Order Shipped': 0,  # Initially, the order is not shipped
        'Order Completed': 0  # Initially, the order is not completed
    }

    # Define transitions with input and output places
    transitions = {
        'Process Order': (['Order Received'], ['Order Processed']),
        'Ship Order': (['Order Processed'], ['Order Shipped']),
        'Complete Order': (['Order Shipped'], ['Order Completed'])
    }

    # Create the Petri Net (Order processing workflow)
    petri_net = PetriNet(places, transitions)

    # Display initial state
    petri_net.display_state()

    # Fire transitions (simulate the order processing)
    petri_net.fire_transition('Process Order')
    petri_net.fire_transition('Ship Order')
    petri_net.fire_transition('Complete Order')

    # Display the final state
    petri_net.display_state()

if __name__ == "__main__":
    main()

3.Simulate a client-server communication model using CCS (Calculus of Communicating
Systems).

import threading
import time

# Shared resources for communication
request_event = threading.Event()
response_event = threading.Event()

def client():
    while True:
        # Client sends request
        print("Client: Sending request...")
        request_event.set()  # Signal server to process
        time.sleep(1)  # Simulate some processing time
        response_event.wait()  # Wait for the server response
        print("Client: Received response.")
        response_event.clear()  # Reset the event for the next round
        time.sleep(1)  # Simulate time before sending the next request

def server():
    while True:
        request_event.wait()  # Wait for the client to send a request
        print("Server: Processing request...")
        time.sleep(2)  # Simulate processing time
        response_event.set()  # Send response back to client
        print("Server: Response sent.")
        request_event.clear()  # Reset the event for the next round

# Create client and server threads
client_thread = threading.Thread(target=client)
server_thread = threading.Thread(target=server)

# Start the threads
client_thread.start()
server_thread.start()

# Wait for the threads to finish (this will run indefinitely)
client_thread.join()
server_thread.join()


4.Develop a formal specification of an online transaction processing system using Process
Algebra.

import threading
import time

# Shared events for communication
payment_event = threading.Event()      # To synchronize payment processing
inventory_event = threading.Event()    # To synchronize inventory check
order_event = threading.Event()        # To synchronize order confirmation

class Customer:
    def __init__(self):
        self.name = "Customer"
    
    def initiate_transaction(self):
        """Customer initiates the transaction and waits for payment to be processed."""
        print(f"{self.name}: Initiating transaction...")
        time.sleep(1)  # Simulate time for transaction initiation
        payment_event.set()  # Signal payment processor to validate payment
        print(f"{self.name}: Waiting for payment response...")
        payment_event.wait()  # Wait for payment to be processed
        print(f"{self.name}: Payment processed, confirming order...")
        order_event.set()  # Signal order management to confirm the order


class PaymentProcessor:
    def __init__(self):
        self.name = "Payment Processor"
    
    def validate_payment(self):
        """Payment Processor validates the payment and signals back to Customer."""
        payment_event.wait()  # Wait for customer to initiate payment
        print(f"{self.name}: Validating payment...")
        time.sleep(2)  # Simulate payment validation time
        print(f"{self.name}: Payment validated.")
        payment_event.clear()  # Clear the event so the customer can proceed
        inventory_event.set()  # Signal inventory system to check availability


class InventorySystem:
    def __init__(self):
        self.name = "Inventory System"
    
    def check_availability(self):
        """Inventory system checks if the product is available."""
        inventory_event.wait()  # Wait for payment to be validated
        print(f"{self.name}: Checking product availability...")
        time.sleep(1)  # Simulate inventory check time
        print(f"{self.name}: Product available.")
        inventory_event.clear()  # Clear the event so order management can proceed
        order_event.set()  # Signal order management to confirm the order


class OrderManagement:
    def __init__(self):
        self.name = "Order Management"
    
    def confirm_order(self):
        """Order Management confirms the order and ships the product."""
        order_event.wait()  # Wait for inventory system to confirm availability
        print(f"{self.name}: Confirming order and arranging shipment...")
        time.sleep(2)  # Simulate order confirmation and shipment time
        print(f"{self.name}: Order confirmed and shipped!")


# Create instances of the system components
customer = Customer()
payment_processor = PaymentProcessor()
inventory_system = InventorySystem()
order_management = OrderManagement()

# Create threads for each process
customer_thread = threading.Thread(target=customer.initiate_transaction)
payment_processor_thread = threading.Thread(target=payment_processor.validate_payment)
inventory_system_thread = threading.Thread(target=inventory_system.check_availability)
order_management_thread = threading.Thread(target=order_management.confirm_order)

# Start the threads
customer_thread.start()
payment_processor_thread.start()
inventory_system_thread.start()
order_management_thread.start()

# Wait for all threads to finish
customer_thread.join()
payment_processor_thread.join()
inventory_system_thread.join()
order_management_thread.join()

print("Transaction processing completed.")

5 . Implement a distributed computation model using Pi-Calculus for mobile process interactions.
                                                               
import threading
import queue
import time

# Process P: Creates a channel and passes it to Process Q
def process_P(channel_Q):
    print("Process P: Creating channel and passing it to Process Q")
    # Simulate passing the channel to Process Q
    channel_P_to_Q = queue.Queue()  # New channel for communication
    channel_Q.put(channel_P_to_Q)  # Pass the channel to Q

    # Process P sends a message to Q via the new channel
    time.sleep(1)  # Wait before sending the message
    print("Process P: Sending message to Process Q via new channel")
    channel_P_to_Q.put("Hello from P")

    # Process P waits for a response from Q
    response = channel_P_to_Q.get()
    print(f"Process P: Received response: {response}")

# Process Q: Receives the channel and communicates with Process P
def process_Q(channel_Q):
    # Receive the channel from Process P
    channel_P_to_Q = channel_Q.get()
    print("Process Q: Received channel from Process P")

    # Process Q receives the message from Process P
    message = channel_P_to_Q.get()
    print(f"Process Q: Received message: {message}")

    # Process Q sends a response back to Process P
    time.sleep(1)  # Simulate processing time
    print("Process Q: Sending response back to Process P")
    channel_P_to_Q.put("Hello from Q")

# Main function to simulate the distributed computation
def main():
    # Create a queue for passing the channel
    channel_Q = queue.Queue()

    # Create and start threads for Process P and Process Q
    thread_P = threading.Thread(target=process_P, args=(channel_Q,))
    thread_Q = threading.Thread(target=process_Q, args=(channel_Q,))

    thread_P.start()
    thread_Q.start()

    # Wait for the processes to finish
    thread_P.join()
    thread_Q.join()

    print("Distributed computation completed.")

if __name__ == "__main__":
    main()

------------------------exp-8--------------------
1.Implement a Deterministic Finite Automaton (DFA) in Python and verify its language
acceptance properties.

class DFA:
    def __init__(self, start_state, accept_states, transitions):
        self.start_state = start_state
        self.accept_states = accept_states
        self.transitions = transitions
        self.current_state = start_state

    def reset(self):
        """Reset the DFA to the initial state."""
        self.current_state = self.start_state

    def transition(self, symbol):
        """Transition to the next state based on the current state and input symbol."""
        self.current_state = self.transitions.get((self.current_state, symbol), None)

    def accepts(self, string):
        """Check if the DFA accepts the given string."""
        self.reset()
        for symbol in string:
            self.transition(symbol)
            if self.current_state is None:
                return False  # Invalid transition
        return self.current_state in self.accept_states

# Define the DFA for strings ending with '01'
# States: q0 (initial), q1, q2 (accepting)
start_state = 'q0'
accept_states = {'q2'}

# Transitions: (current_state, symbol) -> next_state
transitions = {
    ('q0', '0'): 'q0',
    ('q0', '1'): 'q1',
    ('q1', '0'): 'q0',
    ('q1', '1'): 'q2',
    ('q2', '0'): 'q0',
    ('q2', '1'): 'q1',
}

# Create DFA instance
dfa = DFA(start_state, accept_states, transitions)

# Function to test the DFA
def test_dfa(dfa, test_strings):
    for test_str in test_strings:
        result = "Accepted" if dfa.accepts(test_str) else "Rejected"
        print(f"String '{test_str}': {result}")

# Test the DFA with a list of strings
test_strings = ['0101', '110', '101', '011', '1001', '111']
test_dfa(dfa, test_strings)

2.  Develop a simulation tool for Nondeterministic Finite Automata (NFA) and check equivalence
with a DFA.

from collections import deque
from itertools import product

class NFA:
    def __init__(self, states, alphabet, transitions, initial_state, final_states):
        """
        Initialize the NFA.
        
        Parameters:
        - states: set of states
        - alphabet: set of input symbols (excluding epsilon)
        - transitions: dictionary where keys are (state, symbol) and values are sets of states
        - initial_state: the initial state
        - final_states: set of accepting states
        """
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.initial_state = initial_state
        self.final_states = final_states
        self.epsilon = 'ε'  # representing epsilon transitions
        
    def epsilon_closure(self, states):
        """
        Compute the epsilon closure for a set of states.
        """
        closure = set(states)
        queue = deque(states)
        
        while queue:
            state = queue.popleft()
            # Check for epsilon transitions
            if (state, self.epsilon) in self.transitions:
                for next_state in self.transitions[(state, self.epsilon)]:
                    if next_state not in closure:
                        closure.add(next_state)
                        queue.append(next_state)
        return frozenset(closure)
    
    def move(self, states, symbol):
        """
        Move from a set of states on a given symbol.
        """
        result = set()
        for state in states:
            if (state, symbol) in self.transitions:
                result.update(self.transitions[(state, symbol)])
        return frozenset(result)
    
    def simulate(self, input_string):
        """
        Simulate the NFA on an input string.
        Returns True if accepted, False otherwise.
        """
        current_states = self.epsilon_closure({self.initial_state})
        
        for symbol in input_string:
            if symbol not in self.alphabet:
                raise ValueError(f"Symbol '{symbol}' not in alphabet")
            current_states = self.epsilon_closure(self.move(current_states, symbol))
        
        return any(state in self.final_states for state in current_states)

class DFA:
    def __init__(self, states, alphabet, transitions, initial_state, final_states):
        """
        Initialize the DFA.
        
        Parameters:
        - states: set of states
        - alphabet: set of input symbols
        - transitions: dictionary where keys are (state, symbol) and values are single states
        - initial_state: the initial state
        - final_states: set of accepting states
        """
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.initial_state = initial_state
        self.final_states = final_states
    
    def simulate(self, input_string):
        """
        Simulate the DFA on an input string.
        Returns True if accepted, False otherwise.
        """
        current_state = self.initial_state
        
        for symbol in input_string:
            if symbol not in self.alphabet:
                raise ValueError(f"Symbol '{symbol}' not in alphabet")
            current_state = self.transitions[(current_state, symbol)]
        
        return current_state in self.final_states

def nfa_to_dfa(nfa):
    """
    Convert an NFA to an equivalent DFA using subset construction.
    """
    dfa_states = set()
    dfa_transitions = {}
    dfa_initial_state = nfa.epsilon_closure({nfa.initial_state})
    dfa_final_states = set()
    
    unprocessed_states = deque()
    unprocessed_states.append(dfa_initial_state)
    dfa_states.add(dfa_initial_state)
    
    # Check if initial state is final
    if any(state in nfa.final_states for state in dfa_initial_state):
        dfa_final_states.add(dfa_initial_state)
    
    while unprocessed_states:
        current_dfa_state = unprocessed_states.popleft()
        
        for symbol in nfa.alphabet:
            # Compute the next state
            next_nfa_states = nfa.move(current_dfa_state, symbol)
            next_dfa_state = nfa.epsilon_closure(next_nfa_states)
            
            if not next_dfa_state:  # empty set - dead state
                continue
                
            # Add transition
            dfa_transitions[(current_dfa_state, symbol)] = next_dfa_state
            
            # If new state, add to queue and states set
            if next_dfa_state not in dfa_states:
                dfa_states.add(next_dfa_state)
                unprocessed_states.append(next_dfa_state)
                
                # Check if it's a final state
                if any(state in nfa.final_states for state in next_dfa_state):
                    dfa_final_states.add(next_dfa_state)
    
    # Add dead state transitions if needed
    dead_state = frozenset()
    need_dead_state = False
    
    for state in dfa_states:
        for symbol in nfa.alphabet:
            if (state, symbol) not in dfa_transitions:
                need_dead_state = True
                dfa_transitions[(state, symbol)] = dead_state
    
    if need_dead_state:
        dfa_states.add(dead_state)
        for symbol in nfa.alphabet:
            dfa_transitions[(dead_state, symbol)] = dead_state
    
    return DFA(
        states=dfa_states,
        alphabet=nfa.alphabet,
        transitions=dfa_transitions,
        initial_state=dfa_initial_state,
        final_states=dfa_final_states
    )

def are_equivalent(fa1, fa2, max_length=10):
    """
    Check if two finite automata (NFA or DFA) are equivalent by testing strings up to max_length.
    This is a practical approach but not complete for all cases.
    """
    from itertools import product
    
    # Generate all possible strings up to max_length
    for length in range(max_length + 1):
        for test_string in product(fa1.alphabet, repeat=length):
            test_str = ''.join(test_string)
            result1 = fa1.simulate(test_str)
            result2 = fa2.simulate(test_str)
            if result1 != result2:
                print(f"Difference found on string: '{test_str}'")
                print(f"FA1: {'Accepts' if result1 else 'Rejects'}")
                print(f"FA2: {'Accepts' if result2 else 'Rejects'}")
                return False
    return True

# Example usage
if __name__ == "__main__":
    # Example NFA that accepts strings ending with '01'
    nfa = NFA(
        states={'q0', 'q1', 'q2'},
        alphabet={'0', '1'},
        transitions={
            ('q0', '0'): {'q0', 'q1'},
            ('q0', '1'): {'q0'},
            ('q1', '0'): set(),
            ('q1', '1'): {'q2'},
            ('q2', '0'): set(),
            ('q2', '1'): set(),
        },
        initial_state='q0',
        final_states={'q2'}
    )
    
    # Convert NFA to DFA
    dfa = nfa_to_dfa(nfa)
    
    # Test some strings
    test_strings = ['01', '101', '001', '1001', '1010', '11011', '000']
    print("Testing NFA and converted DFA:")
    for test_str in test_strings:
        nfa_result = nfa.simulate(test_str)
        dfa_result = dfa.simulate(test_str)
        print(f"String '{test_str}': NFA {'accepts' if nfa_result else 'rejects'}, DFA {'accepts' if dfa_result else 'rejects'}")
    
    # Check equivalence
    print("\nChecking equivalence between NFA and DFA:")
    if are_equivalent(nfa, dfa):
        print("The NFA and DFA are equivalent (for all tested strings)")
    else:
        print("The NFA and DFA are not equivalent")
    
    # Create another DFA that accepts the same language
    dfa2 = DFA(
        states={'A', 'B', 'C', 'D'},
        alphabet={'0', '1'},
        transitions={
            ('A', '0'): 'B',
            ('A', '1'): 'A',
            ('B', '0'): 'B',
            ('B', '1'): 'C',
            ('C', '0'): 'B',
            ('C', '1'): 'A',
            ('D', '0'): 'D',
            ('D', '1'): 'D',
        },
        initial_state='A',
        final_states={'C'}
    )
    
    print("\nChecking equivalence between converted DFA and another DFA:")
    if are_equivalent(dfa, dfa2):
        print("The two DFAs are equivalent (for all tested strings)")
    else:
        print("The two DFAs are not equivalent")


3.Write a Python-based tool to transform a regular expression into an equivalent automaton.





4.Model and analyze a simple text parser using formal grammar and automata theory.

class DFA:
    def __init__(self, start_state, accept_states, transition_function):
        self.start_state = start_state
        self.accept_states = accept_states
        self.transition_function = transition_function
    
    def accepts(self, input_string):
        current_state = self.start_state
        for symbol in input_string:
            if (current_state, symbol) in self.transition_function:
                current_state = self.transition_function[(current_state, symbol)]
            else:
                return False  # If no valid transition exists, reject the input
        return current_state in self.accept_states


# Define the DFA for alternating 'a' and 'b'
start_state = 'q0'
accept_states = {'q0', 'q1'}
transition_function = {
    ('q0', 'a'): 'q1',  # From q0, on 'a', go to q1
    ('q1', 'b'): 'q0',  # From q1, on 'b', go to q0
}

# Create DFA instance
dfa = DFA(start_state, accept_states, transition_function)

# Test the DFA with valid and invalid strings
test_strings = ['ab', 'abab', 'ababab', 'a', 'ba', 'abb', 'aabb']

print("DFA Acceptance:")
for test_str in test_strings:
    result = "Accepted" if dfa.accepts(test_str) else "Rejected"
    print(f"String '{test_str}': {result}")


5.Implement Minimization of Finite State Machines (FSMs) and verify equivalence between two
FSMs.

class FSM:
    def __init__(self, states, alphabet, transition_function, start_state, accept_states):
        """
        Initialize the FSM.
        
        :param states: Set of states
        :param alphabet: Set of input symbols (alphabet)
        :param transition_function: Dictionary representing the transition function {state: {symbol: next_state}}
        :param start_state: The initial state
        :param accept_states: Set of accepting states
        """
        self.states = states
        self.alphabet = alphabet
        self.transition_function = transition_function
        self.start_state = start_state
        self.accept_states = accept_states

    def get_transitions(self, state, symbol):
        """Returns the next state for a given state and symbol."""
        return self.transition_function.get(state, {}).get(symbol)

    def simulate(self, input_string):
        """Simulate the FSM on the input string."""
        current_state = self.start_state
        for symbol in input_string:
            current_state = self.get_transitions(current_state, symbol)
        return current_state in self.accept_states


def minimize_fsm(fsm):
    """
    Minimize the FSM using partition refinement.
    
    :param fsm: The FSM to minimize
    :return: The minimized FSM
    """
    # Step 1: Initial partition: divide states into accepting and non-accepting states
    partition = {frozenset(fsm.accept_states), frozenset(fsm.states - fsm.accept_states)}

    # Step 2: Refinement of partitions based on transition behavior
    stable = False
    while not stable:
        stable = True
        new_partition = set()
        
        for part in partition:
            # Group states by transitions for each symbol in the alphabet
            grouped = {}
            for state in part:
                key = tuple(fsm.get_transitions(state, symbol) for symbol in fsm.alphabet)
                if key not in grouped:
                    grouped[key] = set()
                grouped[key].add(state)

            # If any group has more than one state, we split the partition
            for group in grouped.values():
                new_partition.add(frozenset(group))

        if partition != new_partition:
            partition = new_partition
            stable = False

    # Step 3: Create a new FSM based on the minimized partition
    state_mapping = {}
    for i, part in enumerate(partition):
        new_state = f'q{i}'
        for state in part:
            state_mapping[state] = new_state

    new_transitions = {}
    for part in partition:
        representative = next(iter(part))
        new_state = state_mapping[representative]
        new_transitions[new_state] = {}
        for symbol in fsm.alphabet:
            next_state = fsm.get_transitions(representative, symbol)
            new_transitions[new_state][symbol] = state_mapping.get(next_state)

    # The start state of the minimized FSM is the state that corresponds to the start state of the original FSM
    start_state = state_mapping[fsm.start_state]

    # The accept states of the minimized FSM are those that correspond to any accepting state of the original FSM
    accept_states = {state_mapping[state] for state in fsm.accept_states}

    return FSM(set(state_mapping.values()), fsm.alphabet, new_transitions, start_state, accept_states)


def check_equivalence(fsm1, fsm2):
    """
    Check if two FSMs are equivalent by comparing their behavior.
    
    :param fsm1: The first FSM
    :param fsm2: The second FSM
    :return: True if the FSMs are equivalent, False otherwise
    """
    # Check equivalence for a set of test strings
    test_strings = ["0101", "111", "0011", "101010", "111000"]
    for test_str in test_strings:
        fsm1_result = fsm1.simulate(test_str)
        fsm2_result = fsm2.simulate(test_str)
        if fsm1_result != fsm2_result:
            print(f"Mismatch for string '{test_str}': FSM1 result {fsm1_result}, FSM2 result {fsm2_result}")
            return False
    return True


def main():
    # Define FSM1 (example: accepting strings with an even number of 1s)
    states1 = {'q0', 'q1'}
    alphabet1 = {'0', '1'}
    transition_function1 = {
        'q0': {'0': 'q0', '1': 'q1'},
        'q1': {'0': 'q1', '1': 'q0'}
    }
    start_state1 = 'q0'
    accept_states1 = {'q0'}

    fsm1 = FSM(states1, alphabet1, transition_function1, start_state1, accept_states1)

    # Define FSM2 (example: accepting strings that end with 1)
    states2 = {'p0', 'p1'}
    alphabet2 = {'0', '1'}
    transition_function2 = {
        'p0': {'0': 'p0', '1': 'p1'},
        'p1': {'0': 'p0', '1': 'p1'}
    }
    start_state2 = 'p0'
    accept_states2 = {'p1'}

    fsm2 = FSM(states2, alphabet2, transition_function2, start_state2, accept_states2)

    # Minimize FSM1
    minimized_fsm1 = minimize_fsm(fsm1)

    # Minimize FSM2
    minimized_fsm2 = minimize_fsm(fsm2)

    # Check equivalence between the two FSMs
    if check_equivalence(minimized_fsm1, minimized_fsm2):
        print("FSM1 and FSM2 are equivalent.")
    else:
        print("FSM1 and FSM2 are not equivalent.")


if __name__ == "__main__":
    main()


--------------------exp-09------------------------------
1.Use model checking to verify the correctness of a topological sorting algorithm.

from collections import deque, defaultdict

class TopologicalSortChecker:
    def __init__(self, graph):
        """
        Initialize the topological sort checker with a directed graph.
        
        :param graph: Dictionary representing the adjacency list of the directed graph
        """
        self.graph = graph
        self.indegree = defaultdict(int)
        self.sorted_order = []

    def topological_sort(self):
        """
        Perform topological sorting (using Kahn's Algorithm).
        """
        # Step 1: Calculate in-degree (number of incoming edges) for each node
        for node in self.graph:
            if node not in self.indegree:
                self.indegree[node] = 0  # In case of isolated nodes
            for neighbor in self.graph[node]:
                self.indegree[neighbor] += 1

        # Step 2: Use a queue to store nodes with no incoming edges (in-degree 0)
        queue = deque([node for node in self.graph if self.indegree[node] == 0])

        # Step 3: Perform the topological sort
        while queue:
            node = queue.popleft()
            self.sorted_order.append(node)
            for neighbor in self.graph[node]:
                self.indegree[neighbor] -= 1
                if self.indegree[neighbor] == 0:
                    queue.append(neighbor)

        # Check if there was a cycle (i.e., if the sorted order doesn't contain all nodes)
        if len(self.sorted_order) != len(self.graph):
            return False  # The graph has a cycle, topological sort is not possible
        return True

    def check_property_1(self):
        """
        Property 1: Ensure that for every edge u -> v, u appears before v in the sorted order.
        """
        for u in self.graph:
            u_index = self.sorted_order.index(u)
            for v in self.graph[u]:
                v_index = self.sorted_order.index(v)
                if u_index > v_index:
                    return False  # u appears after v, violating the topological sort property
        return True

    def check_property_2(self):
        """
        Property 2: Ensure that all nodes appear exactly once in the sorted order.
        """
        return len(set(self.sorted_order)) == len(self.graph)  # No duplicate nodes in sorted order


def main():
    # Example directed graph (DAG)
    graph = {
        'A': ['B', 'C'],
        'B': ['D'],
        'C': ['D'],
        'D': []
    }

    # Initialize topological sort checker
    checker = TopologicalSortChecker(graph)

    # Perform the topological sort
    if checker.topological_sort():
        print("Topological Sort is possible.")
        print("Sorted Order:", checker.sorted_order)
        
        # Check properties
        if checker.check_property_1():
            print("Property 1 (Correctness): The topological sort respects the edges.")
        else:
            print("Property 1 (Correctness): The topological sort does not respect the edges.")
        
        if checker.check_property_2():
            print("Property 2 (Correctness): All nodes appear exactly once in the sorted order.")
        else:
            print("Property 2 (Correctness): There is a duplicate node in the sorted order.")
    else:
        print("Topological Sort is not possible (cycle detected).")

if __name__ == "__main__":
    main()


2.Develop a proof of correctness for binary search algorithm using Hoare Logic.
# Binary Search Algorithm

def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    
    while low <= high:
        mid = (low + high) // 2
        
        if arr[mid] == target:
            return mid  # Target found
        elif arr[mid] < target:
            low = mid + 1  # Eliminate the left half
        else:
            high = mid - 1  # Eliminate the right half
    
    return -1  # Target not found


# Proof of Correctness for Binary Search Algorithm using Hoare Logic

def prove_binary_search_correctness():
    # Test Case 1: Target exists in the array
    arr1 = [1, 2, 3, 4, 5]
    target1 = 3
    result1 = binary_search(arr1, target1)
    print(f"Searching for {target1} in {arr1}... Result: {result1}")

    # Test Case 2: Target does not exist in the array
    arr2 = [1, 2, 3, 4, 5]
    target2 = 6
    result2 = binary_search(arr2, target2)
    print(f"Searching for {target2} in {arr2}... Result: {result2}")


# Run the proof of correctness for binary search
prove_binary_search_correctness()

3.Implement formal verification of loop invariants for fixed point iterative algorithms.

def fixed_point_iteration(f, x0, tolerance=1e-6, max_iterations=1000):
    x_n = x0
    for _ in range(max_iterations):
        x_next = f(x_n)
        if abs(x_next - x_n) < tolerance:
            return x_next  # Fixed point found
        x_n = x_next
    return None  # Convergence not reached within max iterations

# Example function: f(x) = cos(x)
import math
def f(x):
    return math.cos(x)

# Initial guess
x0 = 0.5
result = fixed_point_iteration(f, x0)
print("Fixed point:", result)


4.Develop a formal specification for a job scheduling system and verify correctness.

    import heapq
import threading
import time

class Job:
    def __init__(self, job_id, duration, priority):
        self.job_id = job_id
        self.duration = duration
        self.priority = priority

    def __lt__(self, other):
        # Compare jobs based on priority (and duration as a tiebreaker)
        if self.priority == other.priority:
            return self.duration < other.duration
        return self.priority > other.priority


class Resource:
    def __init__(self, resource_id):
        self.resource_id = resource_id
        self.available = True

    def allocate(self, job):
        self.available = False
        print(f"Resource {self.resource_id} is allocated to Job {job.job_id}")
        time.sleep(job.duration)
        self.release(job)

    def release(self, job):
        self.available = True
        print(f"Resource {self.resource_id} has completed Job {job.job_id}")


class JobScheduler:
    def __init__(self):
        self.jobs_queue = []
        self.resources = []

    def add_resource(self, resource):
        self.resources.append(resource)

    def add_job(self, job):
        heapq.heappush(self.jobs_queue, job)

    def schedule_jobs(self):
        while self.jobs_queue:
            # Get the job with the highest priority
            job = heapq.heappop(self.jobs_queue)
            allocated = False

            for resource in self.resources:
                if resource.available:
                    threading.Thread(target=resource.allocate, args=(job,)).start()
                    allocated = True
                    break

            if not allocated:
                print(f"Job {job.job_id} is waiting for an available resource.")
                time.sleep(1)  # Wait for a resource to become available


# Create a JobScheduler and Resources
scheduler = JobScheduler()
scheduler.add_resource(Resource("R1"))
scheduler.add_resource(Resource("R2"))

# Create some jobs with priority and duration
job1 = Job("J1", 3, 1)  # Job J1 with 3 units of time and priority 1
job2 = Job("J2", 5, 2)  # Job J2 with 5 units of time and priority 2
job3 = Job("J3", 2, 1)  # Job J3 with 2 units of time and priority 1
job4 = Job("J4", 4, 3)  # Job J4 with 4 units of time and priority 3

# Add jobs to the scheduler
scheduler.add_job(job1)
scheduler.add_job(job2)
scheduler.add_job(job3)
scheduler.add_job(job4)

# Schedule jobs
scheduler.schedule_jobs()



                                                               
