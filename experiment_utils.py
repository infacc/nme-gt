import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from qiskit_aer import Aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, random_unitary, random_statevector
from qiskit.circuit.library import ZGate, XGate, IGate, SGate
from qiskit.result import marginal_counts
from tqdm.contrib.concurrent import process_map

def get_nme_state(k):
    norm = 1 / (np.sqrt(1 + k ** 2))
    return Statevector([norm, 0, 0, k * norm])

def tp_circ(cu, resource_state, psi):
    # create teleportation circuit
    qrs = QuantumRegister(2, name='|\psi\\rangle')
    qre = QuantumRegister(2, name='resource state')
    crx = ClassicalRegister(1, name='c_x')
    crz = ClassicalRegister(1, name='c_z')
    crs = ClassicalRegister(2, name='c_psi')
    qc = QuantumCircuit([qrs[0]], qre, [ qrs[1]], crx, crz, crs)
    # initialize state
    qc.initialize(resource_state, qre)
    qc.initialize(psi, qrs)
    qc.cx(qrs[0], qre[0])
    qc.measure(qre[0], crx)
    qc.append(XGate(), [qre[1]]).c_if(crx, 1)
    qc.append(cu, [qre[1], qrs[1]])
    qc.h(qre[1])
    qc.measure(qre[1], crz)
    qc.append(ZGate(), [qrs[0]]).c_if(crz, 1)
    qc.barrier()
    qc.measure(qrs, crs)
    # display(qc.draw(output='mpl'))
    return qc

def comp_circ(g, cu, psi):
    # create compensation circuit
    qrs = QuantumRegister(2, name='|\psi\\rangle')
    qre = QuantumRegister(1, name='resource state')
    crz = ClassicalRegister(1, name='c_z')
    crs = ClassicalRegister(2, name='c_psi')
    qc = QuantumCircuit([qrs[0]], qre, [qrs[1]], crz, crs)
    qc.initialize(psi, qrs)
    qc.initialize([1/np.sqrt(2), 1/np.sqrt(2)], qre)
    qc.append(g, qre)
    qc.append(cu, [qre, qrs[1]])
    qc.h(qre)
    qc.measure(qre, crz)
    qc.append(ZGate(), [qrs[0]]).c_if(crz, 1)
    qc.append(g, [qrs[0]])
    qc.barrier()
    qc.measure(qrs, crs)
    # display(qc.draw(output='mpl'))
    return qc

def get_n_shots(shots, c):
    n_tp_shots = int(shots / (1 + 2 * c))
    n_comp_shots = int(c * n_tp_shots)    
    return n_tp_shots, n_comp_shots

def sample_from_subcircuits(results, shots, weighting_factor):
    n_tp_shots, n_comp_shots = get_n_shots(shots, weighting_factor)

    memory = results[0].get_memory()
    count = Counter(memory[:n_tp_shots])
    counts = defaultdict(int, marginal_counts(count, [2, 3]))
    # counts = marginal_counts(count, [2, 3])

    
    if n_comp_shots > 0:
        # determine qubit indices to marginalize over
        n_qubits = len(next(iter(results[1].get_counts().keys())).split()) + 1
        idx = range(n_qubits - 2, n_qubits)

        for i in range(1, 3):
            memory = results[i].get_memory()
            count = Counter(memory[:n_comp_shots])
            for key, value in marginal_counts(count, idx).items():
                counts[key] += value if i == 1 else -value

    return dict(counts)

def run_experiment(arg_tuple):
    exp_id, controlled_gates, nme, shots, n_rounds, backend = arg_tuple

    exp_list = []
    shots_list = []
    nme_list = []
    round_list = []
    distances_list = []
    exact_res_list = []

    # Loop over controlled gates
    for cu in controlled_gates:
        psi = random_statevector(4)
        res_state = psi.evolve(cu, [0, 1])
        exact_res = res_state.probabilities()

        # Loop over entanglement degrees
        for k in nme:
            weighting_factor = (k-1)**2/(k**2+1)
            nme_state = get_nme_state(k)
            qc_tp = tp_circ(cu, nme_state, psi).decompose(reps=6)
            # qc_comp1 = tp_circ(cu, Statevector([1/2, 1/2, 1/2, 1/2]), psi).decompose(reps=6)
            # qc_comp2 = tp_circ(cu, Statevector([1/2, 1/2j, 1/2j, -1/2]), psi).decompose(reps=6)
            qc_comp1 = comp_circ(IGate(), cu, psi).decompose(reps=6)
            qc_comp2 = comp_circ(SGate(), cu, psi).decompose(reps=6)
            
            n_tp_shots, n_comp_shots = get_n_shots(shots, weighting_factor)

            # Loop over rounds
            for r in range(n_rounds):
                job0 = backend.run(qc_tp, shots=n_tp_shots, memory=True)
                job1 = backend.run(qc_comp1, shots=n_comp_shots, memory=True)
                job2 = backend.run(qc_comp2, shots=n_comp_shots, memory=True)
                results = [job0.result()]
                if n_comp_shots != 0:
                    results.append(job1.result())
                    results.append(job2.result())

                # Loop over shot counts
                for s in range(100, shots + 100, 100):
                    counts = sample_from_subcircuits(results, s, weighting_factor)
                    norm = sum(counts.values())
                    norm_counts = {key: val / norm for key, val in counts.items()}
                    counts_array = np.zeros(4)  
                    for key, value in norm_counts.items():
                        counts_array[int(key, 2)] = value
                    d = np.linalg.norm(exact_res - counts_array)

                    # Append results to lists
                    exp_list.append(exp_id)
                    shots_list.append(s)
                    nme_list.append(k)
                    round_list.append(r)
                    distances_list.append(d)
                    exact_res_list.append(exact_res)

    # Create DataFrame
    df = pd.DataFrame({
        'experiment': exp_list,
        'shots': shots_list,
        'k': nme_list,
        'round': round_list,
        'error': distances_list,
        'exact_res': exact_res_list
    })
    return df

def run_experiments(n_experiments, nme, shots, n_rounds=1, backend=None, u_gates=None, max_workers=10):
    if backend is None:
        backend = Aer.get_backend('qasm_simulator')

    
    if u_gates is None:
        controlled_gates = [random_unitary(2).to_instruction().control(1) for _ in range(10)]
    else:
        controlled_gates = [u.to_instruction().control(1) for u in u_gates]

    r = process_map(run_experiment, [(i, controlled_gates, nme, shots, n_rounds, backend) for i in range(n_experiments)])
            
    df = pd.concat(r)
    return df
