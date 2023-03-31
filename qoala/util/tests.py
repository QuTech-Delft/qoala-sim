from typing import Generator

import netsquid as ns
from netsquid.protocols import Protocol

from pydynaa import EventExpression
from qoala.sim.events import EVENT_WAIT


def yield_from(generator: Generator):
    try:
        while True:
            next(generator)
    except StopIteration as e:
        return e.value


class SimpleNetSquidProtocol(Protocol):
    def __init__(self, generator: Generator) -> None:
        super().__init__("test")
        self._gen = generator
        self.result = None

    def run(self) -> Generator[EventExpression, None, None]:
        self.result = yield from self._gen


def netsquid_run(generator: Generator):
    prot = SimpleNetSquidProtocol(generator)
    prot.start()
    ns.sim_run()
    return prot.result


def netsquid_wait(delta_time: int):
    prot = Protocol()
    prot._schedule_after(delta_time, EVENT_WAIT)
    prot.start()
    ns.sim_run()


def text_equal(text1, text2) -> bool:
    # allows whitespace differences
    lines1 = [line.strip() for line in text1.split("\n") if len(line) > 0]
    lines2 = [line.strip() for line in text2.split("\n") if len(line) > 0]
    for line1, line2 in zip(lines1, lines2):
        if line1 != line2:
            return False
    return True
