from qoala.lang.ehi import EhiBuilder, UnitModule
from qoala.lang.parse import LocalRoutineParser, RequestRoutineParser
from qoala.lang.program import ProgramMeta, QoalaProgram
from qoala.lang.request import RequestRoutine
from qoala.lang.routine import LocalRoutine
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.sim.process import QoalaProcess


def create_local_routine() -> LocalRoutine:
    text = """
SUBROUTINE subrt1
    params: my_value
    returns: m
    uses: 
    keeps:
    request: 
  NETQASM_START
    set Q0 {my_value}
  NETQASM_END
    """

    parsed = LocalRoutineParser(text).parse()
    return parsed["subrt1"]


def create_request_routine() -> RequestRoutine:
    text = """
    REQUEST req1
      callback_type: wait_all
      callback: 
      return_vars: 
      remote_id: 1
      epr_socket_id: 0
      num_pairs: 3
      virt_ids: custom 1, 2, 3
      timeout: 1000
      fidelity: 0.65
      typ: measure_directly
      role: receive
        """
    return RequestRoutineParser(text).parse()["req1"]


def create_process(program: QoalaProgram) -> QoalaProcess:
    ehi = EhiBuilder.perfect_uniform(1, None, [], 0, [], 0)
    unit_module = UnitModule.from_full_ehi(ehi)

    instance = ProgramInstance(
        pid=0,
        program=program,
        inputs=ProgramInput({}),
        unit_module=unit_module,
    )
    mem = ProgramMemory(pid=0)

    process = QoalaProcess(
        prog_instance=instance,
        prog_memory=mem,
        csockets={},
        epr_sockets=program.meta.epr_sockets,
        result=ProgramResult(values={}),
    )
    return process


def test1():
    routine = create_local_routine()

    program = QoalaProgram(
        blocks=[], local_routines={"subrt1": routine}, meta=ProgramMeta.empty("")
    )
    process = create_process(program)

    assert len(process.get_all_local_routines()) == 1
    assert process.get_local_routine("subrt1") == routine
    assert len(process.qnos_mem.get_all_running_local_routines()) == 0


def test2():
    routine = create_request_routine()

    program = QoalaProgram(
        blocks=[], request_routines={"req1": routine}, meta=ProgramMeta.empty("")
    )

    process = create_process(program)
    assert len(process.get_all_request_routines()) == 1
    assert process.get_request_routine("req1") == routine
    assert len(process.qnos_mem.get_all_running_request_routines()) == 0


if __name__ == "__main__":
    test1()
    test2()
