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
        print("\n✅ The expression is correct as per the expected truth table.")
    else:
        print("\n❌ The expression does NOT match the expected truth table.")

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

