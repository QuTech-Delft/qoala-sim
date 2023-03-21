from qoala.lang.ehi import EhiBuilder, UnitModule
from qoala.lang.parse import LocalRoutineParser
from qoala.lang.program import IqoalaProgram, ProgramMeta
from qoala.lang.routine import LocalRoutine
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.runtime.schedule import ProgramTaskList
from qoala.sim.process import IqoalaProcess


def create_local_routine() -> LocalRoutine:
    text = """
SUBROUTINE subrt1
    params: my_value
    returns: M0 -> m
    uses: 
    keeps:
    request: 
  NETQASM_START
    set Q0 {my_value}
  NETQASM_END
    """

    parsed = LocalRoutineParser(text).parse()
    return parsed["subrt1"]


def create_process(program: IqoalaProgram) -> IqoalaProcess:
    ehi = EhiBuilder.perfect_uniform(1, None, [], 0, [], 0)
    unit_module = UnitModule.from_full_ehi(ehi)

    instance = ProgramInstance(
        pid=0,
        program=program,
        inputs=ProgramInput({}),
        tasks=ProgramTaskList.empty(program),
        unit_module=unit_module,
    )
    mem = ProgramMemory(pid=0)

    process = IqoalaProcess(
        prog_instance=instance,
        prog_memory=mem,
        csockets={},
        epr_sockets=program.meta.epr_sockets,
        result=ProgramResult(values={}),
    )
    return process


def test1():
    routine = create_local_routine()

    program = IqoalaProgram(
        blocks=[], local_routines={"subrt1": routine}, meta=ProgramMeta.empty("")
    )
    process = create_process(program)

    assert len(process.get_all_local_routines()) == 1
    assert process.get_local_routine("subrt1") == routine
    assert len(process.qnos_mem.get_all_running_routines()) == 0


if __name__ == "__main__":
    test1()
