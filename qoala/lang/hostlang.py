from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Union

from netqasm.lang.operand import Template

IqoalaValue = Union[int, Template, str]


class IqoalaInstructionType(Enum):
    CC = 0
    CL = auto()
    QC = auto()
    QL = auto()


@dataclass
class IqoalaInstructionSignature:
    typ: IqoalaInstructionType
    duration: int = 0


class IqoalaAttribute:
    def __init__(self, value: IqoalaValue) -> None:
        self._value = value

    @property
    def value(self) -> IqoalaValue:
        return self._value


@dataclass(eq=True, frozen=True)
class IqoalaSharedMemLoc:
    loc: str

    def __str__(self) -> str:
        return str(self.loc)


class IqoalaVector:
    def __init__(self, values: List[str]) -> None:
        self._values = values

    @property
    def values(self) -> List[str]:
        return self._values

    def __str__(self) -> str:
        return f"vec<{','.join(v for v in self.values)}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IqoalaVector):
            return NotImplemented
        return self.values == other.values


class ClassicalIqoalaOp:
    OP_NAME: str = None  # type: ignore
    TYP: IqoalaInstructionType = None  # type: ignore

    def __init__(
        self,
        arguments: Optional[Union[List[str], List[IqoalaVector]]] = None,
        results: Optional[List[str]] = None,
        attributes: Optional[List[IqoalaValue]] = None,
    ) -> None:
        # TODO: support list of strs and vectors
        # currently not needed and confuses mypy
        self._arguments: Union[List[str], List[IqoalaVector]]
        self._results: List[str]
        self._attributes: List[IqoalaValue]

        if arguments is None:
            self._arguments = []  # type: ignore
        else:
            self._arguments = arguments

        if results is None:
            self._results = []
        else:
            self._results = results

        if attributes is None:
            self._attributes = []
        else:
            self._attributes = attributes

    def __str__(self) -> str:
        results = ", ".join(str(r) for r in self.results)
        args = ", ".join(str(a) for a in self.arguments)
        attrs = ", ".join(str(a) for a in self.attributes)
        s = ""
        if len(results) > 0:
            s += f"{results} = "

        s += f"{self.op_name}({args})"

        if len(attrs) > 0:
            s += f" : {attrs}"
        return s

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClassicalIqoalaOp):
            return NotImplemented
        return (
            self.results == other.results
            and self.arguments == other.arguments
            and self.attributes == other.attributes
        )

    @classmethod
    def from_generic_args(
        cls, result: Optional[str], args: List[str], attr: Optional[IqoalaValue]
    ) -> ClassicalIqoalaOp:
        raise NotImplementedError

    @property
    def op_name(self) -> str:
        return self.__class__.OP_NAME  # type: ignore

    @property
    def arguments(self) -> Union[List[str], List[IqoalaVector]]:
        return self._arguments

    @property
    def results(self) -> List[str]:
        return self._results

    @property
    def attributes(self) -> List[IqoalaValue]:
        return self._attributes


class AssignCValueOp(ClassicalIqoalaOp):
    OP_NAME = "assign_cval"
    TYP = IqoalaInstructionType.CL

    def __init__(self, result: str, value: IqoalaValue) -> None:
        super().__init__(results=[result], attributes=[value])

    @classmethod
    def from_generic_args(
        cls, result: Optional[str], args: List[str], attr: Optional[IqoalaValue]
    ):
        assert result is not None
        assert len(args) == 0
        assert attr is not None
        return cls(result, attr)


class SendCMsgOp(ClassicalIqoalaOp):
    OP_NAME = "send_cmsg"
    TYP = IqoalaInstructionType.CC

    def __init__(self, csocket: str, value: str) -> None:
        # args:
        #   csocket (int): ID of csocket
        #   value (str): name of variable holding the value to send
        super().__init__(arguments=[csocket, value])

    @classmethod
    def from_generic_args(
        cls, result: Optional[str], args: List[str], attr: Optional[IqoalaValue]
    ):
        assert result is None
        assert len(args) == 2
        assert attr is None
        return cls(args[0], args[1])


class ReceiveCMsgOp(ClassicalIqoalaOp):
    OP_NAME = "recv_cmsg"
    TYP = IqoalaInstructionType.CC

    def __init__(self, csocket: str, result: str) -> None:
        super().__init__(arguments=[csocket], results=[result])

    @classmethod
    def from_generic_args(
        cls, result: Optional[str], args: List[str], attr: Optional[IqoalaValue]
    ):
        assert result is not None
        assert len(args) == 1
        assert attr is None
        return cls(args[0], result)


class AddCValueOp(ClassicalIqoalaOp):
    OP_NAME = "add_cval_c"
    TYP = IqoalaInstructionType.CL

    def __init__(self, result: str, value0: str, value1: str) -> None:
        super().__init__(arguments=[value0, value1], results=[result])

    @classmethod
    def from_generic_args(
        cls, result: Optional[str], args: List[str], attr: Optional[IqoalaValue]
    ):
        assert result is not None
        assert len(args) == 2
        assert attr is None
        return cls(result, args[0], args[1])


class MultiplyConstantCValueOp(ClassicalIqoalaOp):
    OP_NAME = "mult_const"
    TYP = IqoalaInstructionType.CL

    def __init__(self, result: str, value0: str, const: IqoalaValue) -> None:
        # result = value0 * const
        super().__init__(arguments=[value0], attributes=[const], results=[result])

    @classmethod
    def from_generic_args(
        cls, result: Optional[str], args: List[str], attr: Optional[IqoalaValue]
    ):
        assert result is not None
        assert len(args) == 1
        assert attr is not None
        return cls(result, args[0], attr)


class BitConditionalMultiplyConstantCValueOp(ClassicalIqoalaOp):
    OP_NAME = "bcond_mult_const"
    TYP = IqoalaInstructionType.CL

    def __init__(self, result: str, value0: str, cond: str, const: IqoalaValue) -> None:
        # if const == 1:
        #   result = value0 * const
        # else:
        #   result = value0
        super().__init__(arguments=[value0, cond], attributes=[const], results=[result])

    @classmethod
    def from_generic_args(
        cls, result: Optional[str], args: List[str], attr: Optional[IqoalaValue]
    ):
        assert result is not None
        assert len(args) == 2
        assert attr is not None
        return cls(result, args[0], args[1], attr)


class RunSubroutineOp(ClassicalIqoalaOp):
    OP_NAME = "run_subroutine"
    TYP = IqoalaInstructionType.CL

    def __init__(self, result: IqoalaVector, values: IqoalaVector, subrt: str) -> None:
        super().__init__(results=[result], arguments=[values], attributes=[subrt])

    @classmethod
    def from_generic_args(
        cls, result: Optional[str], args: List[str], attr: Optional[IqoalaValue]
    ):
        if result is not None:
            assert isinstance(result, IqoalaVector)
        assert len(args) == 1
        assert isinstance(args[0], IqoalaVector)
        assert isinstance(attr, str)
        return cls(result, args[0], attr)

    @property
    def subroutine(self) -> str:
        assert isinstance(self.attributes[0], str)
        return self.attributes[0]

    def __str__(self) -> str:
        return super().__str__()


class ReturnResultOp(ClassicalIqoalaOp):
    OP_NAME = "return_result"
    TYP = IqoalaInstructionType.CL

    def __init__(self, value: str) -> None:
        super().__init__(arguments=[value])

    @classmethod
    def from_generic_args(
        cls, result: Optional[str], args: List[str], attr: Optional[IqoalaValue]
    ):
        assert result is None
        assert len(args) == 1
        assert attr is None
        return cls(args[0])


class BasicBlockType(Enum):
    HOST = 0
    LR = auto()
    RR = auto()


@dataclass
class BasicBlockAnnotations:
    deadline: int


@dataclass
class BasicBlock:
    name: str
    typ: BasicBlockType
    instructions: List[ClassicalIqoalaOp]

    def __str__(self) -> str:
        s = f"^{self.name} {{type = {self.typ.name}}}:\n"
        return s + "\n".join("    " + str(i) for i in self.instructions)
