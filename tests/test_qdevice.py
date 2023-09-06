import netsquid as ns
import pytest
from netsquid.components import instructions as ns_instr
from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_CXDIR,
    INSTR_CYDIR,
    INSTR_H,
    INSTR_INIT,
    INSTR_MEASURE,
    INSTR_ROT_X,
    INSTR_ROT_Y,
    INSTR_ROT_Z,
    INSTR_X,
    INSTR_Y,
    INSTR_Z,
)
from netsquid.components.qprocessor import MissingInstructionError
from netsquid.nodes import Node
from netsquid.qubits import ketstates

from qoala.runtime.instructions import (
    INSTR_BICHROMATIC,
    INSTR_MEASURE_ALL,
    INSTR_ROT_X_ALL,
    INSTR_ROT_Y_ALL,
    INSTR_ROT_Z_ALL,
)
from qoala.runtime.lhi import LhiTopologyBuilder
from qoala.sim.build import build_qprocessor_from_topology
from qoala.sim.qdevice import (
    NonInitializedQubitError,
    QDevice,
    QDeviceCommand,
    UnsupportedQDeviceCommandError,
)
from qoala.util.math import PI, PI_OVER_2, has_state
from qoala.util.tests import netsquid_run


def perfect_uniform_qdevice(num_qubits: int) -> QDevice:
    topology = LhiTopologyBuilder.perfect_uniform(
        num_qubits=num_qubits,
        single_instructions=[
            INSTR_INIT,
            INSTR_X,
            INSTR_Y,
            INSTR_Z,
            INSTR_H,
            INSTR_ROT_X,
            INSTR_ROT_Y,
            INSTR_ROT_Z,
            INSTR_MEASURE,
            INSTR_MEASURE_ALL,
        ],
        single_duration=5e3,
        two_instructions=[INSTR_CNOT],
        two_duration=100e3,
        all_qubit_instructions=[
            INSTR_INIT,
            INSTR_MEASURE_ALL,
            INSTR_ROT_X_ALL,
            INSTR_ROT_Y_ALL,
            INSTR_ROT_Z_ALL,
            INSTR_BICHROMATIC,
        ],
        all_qubit_duration=100e3,
    )
    processor = build_qprocessor_from_topology(name="processor", topology=topology)
    node = Node(name="alice", qmemory=processor)
    return QDevice(node=node, topology=topology)


def perfect_nv_star_qdevice(num_qubits: int) -> QDevice:
    topology = LhiTopologyBuilder.perfect_star(
        num_qubits=num_qubits,
        comm_instructions=[
            INSTR_INIT,
            INSTR_ROT_X,
            INSTR_ROT_Y,
            INSTR_ROT_Z,
            INSTR_MEASURE,
        ],
        comm_duration=5e3,
        mem_instructions=[
            INSTR_INIT,
            INSTR_ROT_X,
            INSTR_ROT_Y,
            INSTR_ROT_Z,
        ],
        mem_duration=1e4,
        two_instructions=[INSTR_CXDIR, INSTR_CYDIR],
        two_duration=1e5,
    )
    processor = build_qprocessor_from_topology(name="processor", topology=topology)
    node = Node(name="alice", qmemory=processor)
    return QDevice(node=node, topology=topology)


def test_static_generic():
    num_qubits = 3
    qdevice = perfect_uniform_qdevice(num_qubits)

    assert qdevice.qprocessor.num_positions == num_qubits

    assert qdevice.get_qubit_count() == num_qubits
    assert qdevice.get_comm_qubit_count() == num_qubits
    assert qdevice.get_comm_qubit_ids() == {i for i in range(num_qubits)}
    assert qdevice.get_non_comm_qubit_ids() == set()
    assert qdevice.get_all_qubit_ids() == {i for i in range(num_qubits)}


def test_static_nv():
    num_qubits = 3
    qdevice = perfect_nv_star_qdevice(num_qubits)

    assert qdevice.qprocessor.num_positions == num_qubits
    with pytest.raises(MissingInstructionError):
        qdevice.qprocessor.get_instruction_duration(ns_instr.INSTR_CNOT, [0, 1])

    with pytest.raises(MissingInstructionError):
        qdevice.qprocessor.get_instruction_duration(ns_instr.INSTR_CXDIR, [1, 0])

    # Should not raise error:
    assert (
        qdevice.qprocessor.get_instruction_duration(ns_instr.INSTR_CXDIR, [0, 1]) == 1e5
    )

    assert qdevice.get_qubit_count() == num_qubits
    assert qdevice.get_comm_qubit_count() == 1
    assert qdevice.get_comm_qubit_ids() == {0}
    assert qdevice.get_non_comm_qubit_ids() == {i for i in range(1, num_qubits)}
    assert qdevice.get_all_qubit_ids() == {i for i in range(num_qubits)}


def test_initalize_generic():
    ns.sim_reset()

    num_qubits = 3
    qdevice = perfect_uniform_qdevice(num_qubits)

    # All qubits are not initalized yet.
    assert qdevice.get_local_qubit(0) is None
    assert qdevice.get_local_qubit(1) is None
    assert qdevice.get_local_qubit(2) is None

    # Initialize qubit 0.
    commands = [QDeviceCommand(ns_instr.INSTR_INIT, [0])]
    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    # Qubit 0 should be initalized and have state |0>
    q0 = qdevice.get_local_qubit(0)
    assert has_state(q0, ketstates.s0)

    commands = [QDeviceCommand(ns_instr.INSTR_X, [0])]
    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    # Qubit 0 should be initalized and have state |1>
    q0 = qdevice.get_local_qubit(0)
    assert has_state(q0, ketstates.s1)

    # Qubits 1 and 2 are still not initalized.
    assert qdevice.get_local_qubit(1) is None
    assert qdevice.get_local_qubit(2) is None

    # Initialize qubit 1.
    commands = [QDeviceCommand(ns_instr.INSTR_INIT, [1])]
    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    # Qubit 0 should still be in |1>.
    # Qubit 1 should be in |0>.
    q0 = qdevice.get_local_qubit(0)
    q1 = qdevice.get_local_qubit(1)
    assert has_state(q0, ketstates.s1)
    assert has_state(q1, ketstates.s0)

    # Test getting multiple qubits at the same time.
    [q0, q1, q2] = qdevice.get_local_qubits([0, 1, 2])
    assert has_state(q0, ketstates.s1)
    assert has_state(q1, ketstates.s0)
    assert q2 is None


def test_initalize_nv():
    ns.sim_reset()

    num_qubits = 3
    qdevice = perfect_nv_star_qdevice(num_qubits)

    # All qubits are not initalized yet.
    assert qdevice.get_local_qubit(0) is None
    assert qdevice.get_local_qubit(1) is None
    assert qdevice.get_local_qubit(2) is None

    # Initialize qubit 0.
    commands = [QDeviceCommand(ns_instr.INSTR_INIT, [0])]
    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    # Qubit 0 should be initalized and have state |0>
    q0 = qdevice.get_local_qubit(0)
    assert has_state(q0, ketstates.s0)

    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [QDeviceCommand(ns_instr.INSTR_H, [0])]
        netsquid_run(qdevice.execute_commands(commands))
        ns.sim_run()

    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [QDeviceCommand(ns_instr.INSTR_X, [0])]
        netsquid_run(qdevice.execute_commands(commands))
        ns.sim_run()

    commands = [QDeviceCommand(ns_instr.INSTR_ROT_X, [0], angle=PI)]
    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    q0 = qdevice.get_local_qubit(0)
    assert has_state(q0, ketstates.s1)

    # Qubits 1 and 2 are still not initalized.
    assert qdevice.get_local_qubit(1) is None
    assert qdevice.get_local_qubit(2) is None

    # Initialize qubit 1.
    commands = [QDeviceCommand(ns_instr.INSTR_INIT, [1])]
    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    # Qubit 0 should still be in |1>.
    # Qubit 1 should be in |0>.
    q0 = qdevice.get_local_qubit(0)
    q1 = qdevice.get_local_qubit(1)
    assert has_state(q0, ketstates.s1)
    assert has_state(q1, ketstates.s0)

    # Test getting multiple qubits at the same time.
    [q0, q1, q2] = qdevice.get_local_qubits([0, 1, 2])
    assert has_state(q0, ketstates.s1)
    assert has_state(q1, ketstates.s0)
    assert q2 is None


def test_rotations_generic():
    ns.sim_reset()

    num_qubits = 3
    qdevice = perfect_uniform_qdevice(num_qubits)

    # All qubits are not initalized yet.
    assert qdevice.get_local_qubit(0) is None
    assert qdevice.get_local_qubit(1) is None
    assert qdevice.get_local_qubit(2) is None

    # Initialize qubit 0 and do Y-rotation of PI.
    commands = [
        QDeviceCommand(ns_instr.INSTR_INIT, [0]),
        QDeviceCommand(ns_instr.INSTR_Y, [0]),
    ]
    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    q0 = qdevice.get_local_qubit(0)
    assert has_state(q0, ketstates.s1)

    # Initialize qubit 1 and do H gate.
    commands = [
        QDeviceCommand(ns_instr.INSTR_INIT, [1]),
        QDeviceCommand(ns_instr.INSTR_H, [1]),
    ]
    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    q1 = qdevice.get_local_qubit(1)
    assert has_state(q1, ketstates.h0)

    # Initialize qubit 2, do a H gate, and a Z-rotation of PI/2.
    commands = [
        QDeviceCommand(ns_instr.INSTR_INIT, [2]),
        QDeviceCommand(ns_instr.INSTR_H, [2]),
        QDeviceCommand(ns_instr.INSTR_ROT_Z, [2], angle=PI_OVER_2),
    ]
    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    q2 = qdevice.get_local_qubit(2)
    assert has_state(q2, ketstates.y0)


def test_rotations_nv():
    ns.sim_reset()

    num_qubits = 3
    qdevice = perfect_nv_star_qdevice(num_qubits)

    # All qubits are not initalized yet.
    assert qdevice.get_local_qubit(0) is None
    assert qdevice.get_local_qubit(1) is None
    assert qdevice.get_local_qubit(2) is None

    # NV QDevice does not support X-gate.
    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [
            QDeviceCommand(ns_instr.INSTR_INIT, [0]),
            QDeviceCommand(ns_instr.INSTR_X, [0]),
        ]
        netsquid_run(qdevice.execute_commands(commands))
        ns.sim_run()

    # Initialize qubit 0 and do X-rotation of PI.
    commands = [
        QDeviceCommand(ns_instr.INSTR_INIT, [0]),
        QDeviceCommand(ns_instr.INSTR_ROT_X, [0], angle=PI),
    ]
    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    q0 = qdevice.get_local_qubit(0)
    assert has_state(q0, ketstates.s1)

    # Initialize qubit 1 and do a Y-rotation of PI/2.
    commands = [
        QDeviceCommand(ns_instr.INSTR_INIT, [1]),
        QDeviceCommand(ns_instr.INSTR_ROT_Y, [1], angle=PI_OVER_2),
    ]
    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    q1 = qdevice.get_local_qubit(1)
    assert has_state(q1, ketstates.h0)

    # Initialize qubit 2, do a Y-rotation of PI/2, and a Z-rotation of PI/2.
    commands = [
        QDeviceCommand(ns_instr.INSTR_INIT, [2]),
        QDeviceCommand(ns_instr.INSTR_ROT_Y, [2], angle=PI_OVER_2),
        QDeviceCommand(ns_instr.INSTR_ROT_Z, [2], angle=PI_OVER_2),
    ]
    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    q2 = qdevice.get_local_qubit(2)
    assert has_state(q2, ketstates.y0)


def test_measure_generic():
    num_qubits = 3
    qdevice = perfect_uniform_qdevice(num_qubits)

    # Initialize qubit 0 and measure it.
    commands = [
        QDeviceCommand(ns_instr.INSTR_INIT, [0]),
        QDeviceCommand(ns_instr.INSTR_MEASURE, [0]),
    ]

    meas_outcome = netsquid_run(qdevice.execute_commands(commands))

    q0 = qdevice.get_local_qubit(0)
    # Applying the NetSquid measurement instruction should not discard the qubit.
    # (A QnosProcessor should do this manually!)
    assert q0 is not None
    assert has_state(q0, ketstates.s0)

    assert meas_outcome is not None
    assert meas_outcome == 0

    # Initialize qubit 1, apply X gate, and measure it.
    commands = [
        QDeviceCommand(ns_instr.INSTR_INIT, [1]),
        QDeviceCommand(ns_instr.INSTR_X, [1]),
        QDeviceCommand(ns_instr.INSTR_MEASURE, [1]),
    ]

    meas_outcome = netsquid_run(qdevice.execute_commands(commands))

    q1 = qdevice.get_local_qubit(1)
    # Applying the NetSquid measurement instruction should not discard the qubit.
    # (A QnosProcessor should do this manually!)
    assert q1 is not None
    assert has_state(q1, ketstates.s1)

    assert meas_outcome is not None
    assert meas_outcome == 1

    # Measure qubit 0 again.
    commands = [
        QDeviceCommand(ns_instr.INSTR_MEASURE, [0]),
    ]

    meas_outcome = netsquid_run(qdevice.execute_commands(commands))

    q0 = qdevice.get_local_qubit(0)
    assert q0 is not None
    assert has_state(q0, ketstates.s0)

    assert meas_outcome is not None
    assert meas_outcome == 0


def test_measure_nv():
    num_qubits = 3
    qdevice = perfect_nv_star_qdevice(num_qubits)

    # Initialize qubit 0 and measure it.
    commands = [
        QDeviceCommand(ns_instr.INSTR_INIT, [0]),
        QDeviceCommand(ns_instr.INSTR_MEASURE, [0]),
    ]

    meas_outcome = netsquid_run(qdevice.execute_commands(commands))

    q0 = qdevice.get_local_qubit(0)
    # Applying the NetSquid measurement instruction should not discard the qubit.
    # (A QnosProcessor should do this manually!)
    assert q0 is not None
    assert has_state(q0, ketstates.s0)

    assert meas_outcome is not None
    assert meas_outcome == 0

    # Initialize qubit 1, and try to measure it. Should raise an error since
    # only qubit 0 can be measured.
    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [
            QDeviceCommand(ns_instr.INSTR_INIT, [1]),
            QDeviceCommand(ns_instr.INSTR_MEASURE, [1]),
        ]
        meas_outcome = netsquid_run(qdevice.execute_commands(commands))

    # Measure qubit 0 again.
    commands = [QDeviceCommand(ns_instr.INSTR_MEASURE, [0])]
    meas_outcome = netsquid_run(qdevice.execute_commands(commands))

    q0 = qdevice.get_local_qubit(0)
    assert q0 is not None
    assert has_state(q0, ketstates.s0)

    assert meas_outcome is not None
    assert meas_outcome == 0


def test_two_qubit_gates_generic():
    ns.sim_reset()

    num_qubits = 3
    qdevice = perfect_uniform_qdevice(num_qubits)

    # All qubits are not initalized yet.
    assert qdevice.get_local_qubit(0) is None
    assert qdevice.get_local_qubit(1) is None
    assert qdevice.get_local_qubit(2) is None

    # Initialize qubits 0 and 2 and apply a CNOT
    commands = [
        QDeviceCommand(ns_instr.INSTR_INIT, [0]),
        QDeviceCommand(ns_instr.INSTR_INIT, [2]),
        QDeviceCommand(ns_instr.INSTR_CNOT, [0, 2]),
    ]
    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    [q0, q1, q2] = qdevice.get_local_qubits([0, 1, 2])
    assert has_state(q0, ketstates.s0)
    assert q1 is None
    assert has_state(q2, ketstates.s0)

    # Apply an X to qubit 2, and a do CNOT between 2 and 0.
    commands = [
        QDeviceCommand(ns_instr.INSTR_X, [2]),
        QDeviceCommand(ns_instr.INSTR_CNOT, [2, 0]),
    ]
    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    [q0, q1, q2] = qdevice.get_local_qubits([0, 1, 2])
    assert has_state(q0, ketstates.s1)
    assert q1 is None
    assert has_state(q2, ketstates.s1)


def test_two_qubit_gates_nv():
    ns.sim_reset()

    num_qubits = 3
    qdevice = perfect_nv_star_qdevice(num_qubits)

    # All qubits are not initalized yet.
    assert qdevice.get_local_qubit(0) is None
    assert qdevice.get_local_qubit(1) is None
    assert qdevice.get_local_qubit(2) is None

    # Initialize qubits 0 and 2 and apply a CXDIR
    commands = [
        QDeviceCommand(ns_instr.INSTR_INIT, [0]),
        QDeviceCommand(ns_instr.INSTR_INIT, [2]),
        QDeviceCommand(ns_instr.INSTR_CXDIR, [0, 2], angle=PI_OVER_2),
    ]
    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    # q0 is |0> so q2 was rotated with pi/2 around X
    [q0, q1, q2] = qdevice.get_local_qubits([0, 1, 2])
    assert has_state(q0, ketstates.s0)
    assert q1 is None
    assert has_state(q2, ketstates.y1)

    # Apply an X to qubit 0, and a do CNOT between 0 and 2.
    commands = [
        QDeviceCommand(ns_instr.INSTR_ROT_X, [0], angle=PI),
        QDeviceCommand(ns_instr.INSTR_CXDIR, [0, 2], angle=PI_OVER_2),
    ]
    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    # q0 is |1> so q2 was rotated with -pi/2 around X
    [q0, q1, q2] = qdevice.get_local_qubits([0, 1, 2])
    assert has_state(q0, ketstates.s1)
    assert q1 is None
    assert has_state(q2, ketstates.s0)


def test_unsupported_commands_generic():
    num_qubits = 3
    qdevice = perfect_uniform_qdevice(num_qubits)

    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [QDeviceCommand(ns_instr.INSTR_T, [0])]
        netsquid_run(qdevice.execute_commands(commands))


def test_unsupported_commands_nv():
    num_qubits = 3
    qdevice = perfect_nv_star_qdevice(num_qubits)

    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [QDeviceCommand(ns_instr.INSTR_X, [0])]
        netsquid_run(qdevice.execute_commands(commands))

    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [QDeviceCommand(ns_instr.INSTR_Y, [0])]
        netsquid_run(qdevice.execute_commands(commands))

    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [QDeviceCommand(ns_instr.INSTR_Z, [0])]
        netsquid_run(qdevice.execute_commands(commands))

    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [QDeviceCommand(ns_instr.INSTR_H, [0])]
        netsquid_run(qdevice.execute_commands(commands))

    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [QDeviceCommand(ns_instr.INSTR_X, [1])]
        netsquid_run(qdevice.execute_commands(commands))

    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [QDeviceCommand(ns_instr.INSTR_MEASURE, [1])]
        netsquid_run(qdevice.execute_commands(commands))

    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [QDeviceCommand(ns_instr.INSTR_MEASURE, [2])]
        netsquid_run(qdevice.execute_commands(commands))

    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [QDeviceCommand(ns_instr.INSTR_CXDIR, [0, 0])]
        netsquid_run(qdevice.execute_commands(commands))

    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [QDeviceCommand(ns_instr.INSTR_CXDIR, [1, 0])]
        netsquid_run(qdevice.execute_commands(commands))

    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [QDeviceCommand(ns_instr.INSTR_CXDIR, [1, 2])]
        netsquid_run(qdevice.execute_commands(commands))


def test_non_initalized():
    num_qubits = 3
    qdevice = perfect_uniform_qdevice(num_qubits)

    with pytest.raises(NonInitializedQubitError):
        commands = [QDeviceCommand(ns_instr.INSTR_X, [0])]
        netsquid_run(qdevice.execute_commands(commands))

    with pytest.raises(NonInitializedQubitError):
        commands = [QDeviceCommand(ns_instr.INSTR_INIT, [0])]
        commands = [QDeviceCommand(ns_instr.INSTR_CNOT, [0, 1])]
        netsquid_run(qdevice.execute_commands(commands))


def test_all_qubit_gates():
    num_qubits = 3
    qdevice = perfect_uniform_qdevice(num_qubits)

    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [QDeviceCommand(ns_instr.INSTR_X)]
        netsquid_run(qdevice.execute_commands(commands))

    with pytest.raises(UnsupportedQDeviceCommandError):
        commands = [QDeviceCommand(INSTR_ROT_X_ALL, [0], angle=PI)]
        netsquid_run(qdevice.execute_commands(commands))

    with pytest.raises(UnsupportedQDeviceCommandError):
        # All qubit commands should not have qubit indices.
        commands = [QDeviceCommand(INSTR_ROT_X_ALL, [0, 1, 2], angle=PI)]
        netsquid_run(qdevice.execute_commands(commands))


def test_initialize_all():
    num_qubits = 3
    qdevice = perfect_uniform_qdevice(num_qubits)

    commands = [
        QDeviceCommand(
            INSTR_INIT,
        )
    ]

    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    q0 = qdevice.get_local_qubit(0)
    q1 = qdevice.get_local_qubit(1)
    q2 = qdevice.get_local_qubit(2)
    assert q0 is not None
    assert q1 is not None
    assert q2 is not None


def test_measure_all():
    num_qubits = 3
    qdevice = perfect_uniform_qdevice(num_qubits)

    commands = [
        QDeviceCommand(
            INSTR_INIT,
        ),
        QDeviceCommand(INSTR_Y, [1]),
        QDeviceCommand(
            INSTR_MEASURE_ALL,
        ),
    ]

    meas_outcome = netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    q0 = qdevice.get_local_qubit(0)
    q1 = qdevice.get_local_qubit(1)
    q2 = qdevice.get_local_qubit(2)
    assert has_state(q0, ketstates.s0)
    assert has_state(q1, ketstates.s1)
    assert has_state(q2, ketstates.s0)

    assert meas_outcome == [0, 1, 0]


def test_rotate_all():
    num_qubits = 3
    qdevice = perfect_uniform_qdevice(num_qubits)

    commands = [
        QDeviceCommand(
            INSTR_INIT,
        ),
        QDeviceCommand(INSTR_ROT_X_ALL, angle=PI),
    ]

    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    q0 = qdevice.get_local_qubit(0)
    q1 = qdevice.get_local_qubit(1)
    q2 = qdevice.get_local_qubit(2)
    assert has_state(q0, ketstates.s1)
    assert has_state(q1, ketstates.s1)
    assert has_state(q2, ketstates.s1)

    commands = [
        QDeviceCommand(
            INSTR_INIT,
        ),
        QDeviceCommand(INSTR_ROT_Y_ALL, angle=PI_OVER_2),
    ]

    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    q0 = qdevice.get_local_qubit(0)
    q1 = qdevice.get_local_qubit(1)
    q2 = qdevice.get_local_qubit(2)
    assert has_state(q0, ketstates.h0)
    assert has_state(q1, ketstates.h0)
    assert has_state(q2, ketstates.h0)

    commands = [
        QDeviceCommand(
            INSTR_INIT,
        ),
        QDeviceCommand(INSTR_ROT_Y_ALL, angle=PI_OVER_2),
        QDeviceCommand(INSTR_ROT_Z_ALL, angle=PI_OVER_2),
    ]

    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    q0 = qdevice.get_local_qubit(0)
    q1 = qdevice.get_local_qubit(1)
    q2 = qdevice.get_local_qubit(2)
    assert has_state(q0, ketstates.y0)
    assert has_state(q1, ketstates.y0)
    assert has_state(q2, ketstates.y0)


def test_bichromatic():
    num_qubits = 3
    qdevice = perfect_uniform_qdevice(num_qubits)

    commands = [
        QDeviceCommand(
            INSTR_INIT,
        ),
        QDeviceCommand(INSTR_ROT_X_ALL, angle=PI),
    ]

    netsquid_run(qdevice.execute_commands(commands))
    ns.sim_run()

    q0 = qdevice.get_local_qubit(0)
    q1 = qdevice.get_local_qubit(1)
    q2 = qdevice.get_local_qubit(2)
    assert q0 is not None
    assert q1 is not None
    assert q2 is not None


def test_is_allowed():
    num_qubits = 3
    qdevice_gen = perfect_uniform_qdevice(num_qubits)
    qdevice_nv = perfect_nv_star_qdevice(num_qubits)

    assert all(
        qdevice_gen.is_allowed(QDeviceCommand(ns_instr.INSTR_INIT, [i]))
        for i in range(num_qubits)
    )
    assert all(
        qdevice_nv.is_allowed(QDeviceCommand(ns_instr.INSTR_INIT, [i]))
        for i in range(num_qubits)
    )

    assert all(
        qdevice_gen.is_allowed(QDeviceCommand(ns_instr.INSTR_X, [i]))
        for i in range(num_qubits)
    )

    assert not any(
        qdevice_nv.is_allowed(QDeviceCommand(ns_instr.INSTR_X, [i]))
        for i in range(num_qubits)
    )

    assert not any(
        qdevice_gen.is_allowed(QDeviceCommand(ns_instr.INSTR_S, [i]))
        for i in range(num_qubits)
    )

    assert not any(
        qdevice_nv.is_allowed(QDeviceCommand(ns_instr.INSTR_S, [i]))
        for i in range(num_qubits)
    )

    # 5 is not in topology
    assert not qdevice_gen.is_allowed(QDeviceCommand(ns_instr.INSTR_X, [5]))


if __name__ == "__main__":
    test_static_generic()
    test_static_nv()
    test_initalize_generic()
    test_initalize_nv()
    test_rotations_generic()
    test_rotations_nv()
    test_measure_generic()
    test_measure_nv()
    test_two_qubit_gates_generic()
    test_two_qubit_gates_nv()
    test_unsupported_commands_generic()
    test_unsupported_commands_nv()
    test_non_initalized()
    test_all_qubit_gates()
    test_initialize_all()
    test_measure_all()
    test_rotate_all()
    test_bichromatic()
    test_is_allowed()
