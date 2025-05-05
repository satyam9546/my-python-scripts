----------------------------exp-5---------------
1.Implement a Kripke Structure in Python and verify Computation Tree Logic (CTL)
properties.

class KripkeStructure:
    def __init__(self, states, transitions, labeling):
        """
        Initialize the Kripke structure.
        
        :param states: Set of states
        :param transitions: Dictionary of transitions {state: [next_state, ...]}
        :param labeling: Dictionary of labeling {state: set of atomic propositions}
        """
        self.states = states
        self.transitions = transitions
        self.labeling = labeling

    def get_successors(self, state):
        """Returns the set of states reachable from the given state."""
        return self.transitions.get(state, [])

    def get_labeling(self, state):
        """Returns the atomic propositions assigned to the state."""
        return self.labeling.get(state, set())

# CTL Formula Evaluation
def evaluate_ctl_formula(ks, state, formula):
    """Evaluates the CTL formula on the Kripke structure at the given state."""
    if formula == "True":
        return True
    elif formula == "False":
        return False
    elif formula.startswith("p"):  # Atomic proposition p
        return "p" in ks.get_labeling(state)
    elif formula.startswith("AX"):  # AX phi (Always in the next state)
        subformula = formula[3:]
        successors = ks.get_successors(state)
        return all(evaluate_ctl_formula(ks, successor, subformula) for successor in successors)
    elif formula.startswith("EX"):  # EX phi (Exists in the next state)
        subformula = formula[3:]
        successors = ks.get_successors(state)
        return any(evaluate_ctl_formula(ks, successor, subformula) for successor in successors)
    elif formula.startswith("AF"):  # AF phi (Always eventually)
        subformula = formula[3:]
        successors = ks.get_successors(state)
        return all(evaluate_ctl_formula(ks, successor, subformula) for successor in successors)
    elif formula.startswith("EF"):  # EF phi (Exists eventually)
        subformula = formula[3:]
        successors = ks.get_successors(state)
        return any(evaluate_ctl_formula(ks, successor, subformula) for successor in successors)
    elif formula.startswith("A"):  # A phi (For all paths, phi)
        subformula = formula[1:]
        successors = ks.get_successors(state)
        return all(evaluate_ctl_formula(ks, successor, subformula) for successor in successors)
    elif formula.startswith("E"):  # E phi (There exists a path, phi)
        subformula = formula[1:]
        successors = ks.get_successors(state)
        return any(evaluate_ctl_formula(ks, successor, subformula) for successor in successors)
    else:
        raise ValueError(f"Unknown formula: {formula}")

def main():
    # Define the Kripke Structure
    states = {'s0', 's1', 's2'}
    transitions = {
        's0': ['s1'],
        's1': ['s2'],
        's2': ['s0'],
    }
    labeling = {
        's0': {'p'},
        's1': {'q'},
        's2': {'p', 'q'},
    }

    ks = KripkeStructure(states, transitions, labeling)

    # Define the CTL formula
    formula = "AXp"  # Always in the next state, "p"
    result = evaluate_ctl_formula(ks, 's0', formula)
    print(f"Result of evaluating '{formula}' at s0: {result}")

    formula = "EXp"  # Exists in the next state, "p"
    result = evaluate_ctl_formula(ks, 's0', formula)
    print(f"Result of evaluating '{formula}' at s0: {result}")

    formula = "AFp"  # Always eventually, "p"
    result = evaluate_ctl_formula(ks, 's0', formula)
    print(f"Result of evaluating '{formula}' at s0: {result}")

if __name__ == "__main__":
    main()

