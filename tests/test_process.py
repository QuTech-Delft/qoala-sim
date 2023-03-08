import pytest

from qoala.lang.ehi import EhiBuilder, UnitModule
from qoala.lang.parse import LocalRoutineParser
from qoala.lang.program import IqoalaProgram, ProgramMeta
from qoala.lang.routine import LocalRoutine
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.runtime.schedule import ProgramTaskList
from qoala.sim.memory import ProgramMemory
from qoala.sim.process import IqoalaProcess
from qoala.util.tests import text_equal


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
    instance = ProgramInstance(
        pid=0,
        program=program,
        inputs=ProgramInput({}),
        tasks=ProgramTaskList.empty(program),
    )
    ehi = EhiBuilder.perfect_uniform(1, None, [], 0, [], 0)
    unit_module = UnitModule.from_full_ehi(ehi)
    mem = ProgramMemory(pid=0, unit_module=unit_module)

    process = IqoalaProcess(
        prog_instance=instance,
        prog_memory=mem,
        csockets={},
        epr_sockets=program.meta.epr_sockets,
        result=ProgramResult(values={}),
        active_routines={},
    )
    return process


def test1():
    routine = create_local_routine()

    program = IqoalaProgram(
        instructions=[], local_routines={"subrt1": routine}, meta=ProgramMeta.empty("")
    )
    process = create_process(program)

    assert len(process.get_all_local_routines()) == 1
    assert process.get_local_routine("subrt1") == routine
    assert len(process.get_all_active_routines()) == 0

    with pytest.raises(KeyError):
        process.instantiate_routine("subrt1", {"hello": 3})

    process.instantiate_routine("subrt1", {"my_value": 3})
    assert len(process.get_all_local_routines()) == 1
    assert process.get_local_routine("subrt1") == routine
    assert len(process.get_all_active_routines()) == 1

    instantiated = process.get_active_routine("subrt1")
    # instance should have a deepcopy of original routine
    assert instantiated.routine != routine

    assert text_equal(str(instantiated.routine.subroutine.instructions[0]), "set Q0 3")


if __name__ == "__main__":
    test1()
