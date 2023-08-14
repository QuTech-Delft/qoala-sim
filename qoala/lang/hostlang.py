from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Union

from netqasm.lang.operand import Template

IqoalaValue = Union[int, Template, str]


class HostLanguageSyntaxError(Exception):
    pass


class IqoalaInstructionType(Enum):
    CC = 0
    CL = auto()
    QC = auto()
    QL = auto()


class IqoalaVar:
    pass


@dataclass(frozen=True)
class IqoalaSingleton(IqoalaVar):
    name: str

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class IqoalaTuple(IqoalaVar):
    values: List[str]

    def __str__(self) -> str:
        return f"tuple<{','.join(v for v in self.values)}>"


@dataclass(frozen=True)
class IqoalaVector(IqoalaVar):
    name: str
    size: IqoalaValue

    def __str__(self) -> str:
        return f"{self.name}<{self.size}>"


@dataclass(frozen=True)
class IqoalaVectorElement(IqoalaVar):
    name: str
    index: int


class ClassicalIqoalaOp:
    OP_NAME: str = None  # type: ignore
    TYP: IqoalaInstructionType = None  # type: ignore

    def __init__(
        self,
        arguments: Optional[List[IqoalaVar]] = None,
        results: Optional[IqoalaVar] = None,
        attributes: Optional[List[IqoalaValue]] = None,
    ) -> None:
        # TODO: support list of strs and tuples
        # currently not needed and confuses mypy
        self._arguments: List[IqoalaVar]
        self._results: IqoalaVar
        self._attributes: List[IqoalaValue]

        if arguments is None:
            self._arguments = []  # type: ignore
        else:
            self._arguments = arguments

        self._results = results  # type: ignore

        if attributes is None:
            self._attributes = []
        else:
            self._attributes = attributes

    def __str__(self) -> str:
        if self.results is None:
            results = ""
        else:
            results = str(self.results)
        # not to write  for the empty tuple
        if self.arguments == [IqoalaTuple([])]:
            args = ""
        else:
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
        cls,
        result: Optional[IqoalaVar],
        args: List[IqoalaVar],
        attr: Optional[IqoalaValue],
    ) -> ClassicalIqoalaOp:
        raise NotImplementedError

    @property
    def op_name(self) -> str:
        return self.__class__.OP_NAME  # type: ignore

    @property
    def arguments(self) -> List[IqoalaVar]:
        return self._arguments

    @property
    def results(self) -> IqoalaVar:
        return self._results

    @property
    def attributes(self) -> List[IqoalaValue]:
        return self._attributes


class AssignCValueOp(ClassicalIqoalaOp):
    OP_NAME = "assign_cval"
    TYP = IqoalaInstructionType.CL

    def __init__(self, result: IqoalaSingleton, value: IqoalaValue) -> None:
        super().__init__(results=result, attributes=[value])

    @classmethod
    def from_generic_args(
        cls,
        result: Optional[IqoalaVar],
        args: List[IqoalaVar],
        attr: Optional[IqoalaValue],
    ):
        if result is None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation must have a result."
            )
        if not isinstance(result, IqoalaSingleton):
            raise HostLanguageSyntaxError
        if len(args) != 0:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation takes 0 arguments but got {len(args)}."
            )
        if attr is None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation must have an attribute."
            )
        return cls(result, attr)


class BusyOp(ClassicalIqoalaOp):
    OP_NAME = "busy"
    TYP = IqoalaInstructionType.CL

    def __init__(self, value: IqoalaValue) -> None:
        super().__init__(attributes=[value])

    @classmethod
    def from_generic_args(
        cls,
        result: Optional[IqoalaVar],
        args: Union[List[IqoalaVar]],
        attr: Optional[IqoalaValue],
    ):
        if result is not None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation cannot have a result."
            )
        if len(args) != 0:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation takes 0 arguments but got {len(args)}."
            )
        if attr is None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation must have an attribute."
            )
        return cls(attr)


class SendCMsgOp(ClassicalIqoalaOp):
    OP_NAME = "send_cmsg"
    TYP = IqoalaInstructionType.CC

    def __init__(
        self,
        csocket: Union[IqoalaSingleton, IqoalaVectorElement],
        value: Union[IqoalaSingleton, IqoalaVectorElement],
    ) -> None:
        # args:
        #   csocket (int): ID of csocket
        #   value (str): name of variable holding the value to send
        super().__init__(arguments=[csocket, value])

    @classmethod
    def from_generic_args(
        cls,
        result: Optional[IqoalaVar],
        args: Union[List[IqoalaVar]],
        attr: Optional[IqoalaValue],
    ):
        if result is not None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation cannot have a result."
            )
        if len(args) != 2:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation takes 2 arguments but got {len(args)}."
            )
        if attr is not None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation cannot have an attribute."
            )
        if (
            not isinstance(args[0], IqoalaSingleton)
            and not isinstance(args[0], IqoalaVectorElement)
        ) or (
            not isinstance(args[1], IqoalaSingleton)
            and not isinstance(args[1], IqoalaVectorElement)
        ):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation arguments must be strings or vector elements."
            )
        return cls(args[0], args[1])


class ReceiveCMsgOp(ClassicalIqoalaOp):
    OP_NAME = "recv_cmsg"
    TYP = IqoalaInstructionType.CC

    def __init__(
        self,
        csocket: Union[IqoalaSingleton, IqoalaVectorElement],
        result: IqoalaSingleton,
    ) -> None:
        super().__init__(arguments=[csocket], results=result)

    @classmethod
    def from_generic_args(
        cls,
        result: Optional[IqoalaVar],
        args: List[IqoalaVar],
        attr: Optional[IqoalaValue],
    ):
        if result is None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation must have a result."
            )
        if not isinstance(result, IqoalaSingleton):
            raise HostLanguageSyntaxError
        if len(args) != 1:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation takes 1 argument but got {len(args)}."
            )
        if attr is not None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation cannot have an attribute."
            )
        if not isinstance(args[0], IqoalaSingleton) and not isinstance(
            args[0], IqoalaVectorElement
        ):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation arguments must be strings or vector elements."
            )
        return cls(args[0], result)


class AddCValueOp(ClassicalIqoalaOp):
    OP_NAME = "add_cval_c"
    TYP = IqoalaInstructionType.CL

    def __init__(
        self,
        result: IqoalaSingleton,
        value0: Union[IqoalaSingleton, IqoalaVectorElement],
        value1: Union[IqoalaSingleton, IqoalaVectorElement],
    ) -> None:
        super().__init__(arguments=[value0, value1], results=result)

    @classmethod
    def from_generic_args(
        cls,
        result: Optional[IqoalaVar],
        args: List[IqoalaVar],
        attr: Optional[IqoalaValue],
    ):
        if result is None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation must have a result."
            )
        if not isinstance(result, IqoalaSingleton):
            raise HostLanguageSyntaxError
        if len(args) != 2:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation takes 2 arguments but got {len(args)}."
            )
        if attr is not None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation cannot have an attribute."
            )
        if (
            not isinstance(args[0], IqoalaSingleton)
            and not isinstance(args[0], IqoalaVectorElement)
        ) or (
            not isinstance(args[1], IqoalaSingleton)
            and not isinstance(args[1], IqoalaVectorElement)
        ):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation arguments must be strings or vector elements."
            )
        return cls(result, args[0], args[1])


class MultiplyConstantCValueOp(ClassicalIqoalaOp):
    OP_NAME = "mult_const"
    TYP = IqoalaInstructionType.CL

    def __init__(
        self,
        result: IqoalaSingleton,
        value0: Union[IqoalaSingleton, IqoalaVectorElement],
        const: IqoalaValue,
    ) -> None:
        # result = value0 * const
        super().__init__(arguments=[value0], attributes=[const], results=result)

    @classmethod
    def from_generic_args(
        cls,
        result: Optional[IqoalaVar],
        args: List[IqoalaVar],
        attr: Optional[IqoalaValue],
    ):
        if result is None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation must have a result."
            )
        if not isinstance(result, IqoalaSingleton):
            raise HostLanguageSyntaxError
        if len(args) != 1:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation takes 1 argument but got {len(args)}."
            )
        if attr is None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation must have an attribute."
            )
        if not isinstance(args[0], IqoalaSingleton) and not isinstance(
            args[0], IqoalaVectorElement
        ):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation arguments must be strings or vector elements."
            )
        return cls(result, args[0], attr)


class BitConditionalMultiplyConstantCValueOp(ClassicalIqoalaOp):
    OP_NAME = "bcond_mult_const"
    TYP = IqoalaInstructionType.CL

    def __init__(
        self,
        result: IqoalaSingleton,
        value0: Union[IqoalaSingleton, IqoalaVectorElement],
        cond: Union[IqoalaSingleton, IqoalaVectorElement],
        const: int,
    ) -> None:
        # if const == 1:
        #   result = value0 * const
        # else:
        #   result = value0
        super().__init__(arguments=[value0, cond], attributes=[const], results=result)

    @classmethod
    def from_generic_args(
        cls,
        result: Optional[IqoalaVar],
        args: List[IqoalaVar],
        attr: Optional[IqoalaValue],
    ):
        if result is None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation must have a result."
            )
        if not isinstance(result, IqoalaSingleton):
            raise HostLanguageSyntaxError
        if len(args) != 2:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation takes 2 arguments but got {len(args)}."
            )
        if attr is None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation must have an attribute."
            )
        if (
            not isinstance(args[0], IqoalaSingleton)
            and not isinstance(args[0], IqoalaVectorElement)
        ) or (
            not isinstance(args[1], IqoalaSingleton)
            and not isinstance(args[1], IqoalaVectorElement)
        ):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation arguments must be strings or vector elements."
            )
        if not isinstance(attr, int):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation attribute must be an integer."
            )
        return cls(result, args[0], args[1], attr)


class RunSubroutineOp(ClassicalIqoalaOp):
    OP_NAME = "run_subroutine"
    TYP = IqoalaInstructionType.CL

    def __init__(
        self,
        result: Optional[Union[IqoalaTuple, IqoalaVector]],
        values: IqoalaTuple,
        subrt: str,
    ) -> None:
        super().__init__(results=result, arguments=[values], attributes=[subrt])

    @classmethod
    def from_generic_args(
        cls,
        result: Optional[IqoalaVar],
        args: List[IqoalaVar],
        attr: Optional[IqoalaValue],
    ):
        if result is not None:
            if not isinstance(result, IqoalaTuple) and not isinstance(
                result, IqoalaVector
            ):
                raise HostLanguageSyntaxError(
                    f"{cls.OP_NAME} operation cannot have a result of type {type(result)}. "
                    f"It must be either IqoalaTuple or IqoalaVector."
                )
        if len(args) == 0:
            args = [IqoalaTuple([])]

        if len(args) != 1:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation takes 1 argument but got {len(args)}."
            )
        if not isinstance(args[0], IqoalaTuple):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation argument must be an IqoalaTuple."
            )
        if attr is None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation must have an attribute."
            )
        if not isinstance(attr, str):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation attribute must be a string."
            )

        return cls(result, args[0], attr)

    @property
    def subroutine(self) -> str:
        assert isinstance(self.attributes[0], str)
        return self.attributes[0]

    def __str__(self) -> str:
        return super().__str__()


class RunRequestOp(ClassicalIqoalaOp):
    OP_NAME = "run_request"
    TYP = IqoalaInstructionType.CL

    def __init__(
        self,
        result: Optional[Union[IqoalaTuple, IqoalaVector]],
        values: IqoalaTuple,
        routine: str,
    ) -> None:
        super().__init__(results=result, arguments=[values], attributes=[routine])

    @classmethod
    def from_generic_args(
        cls,
        result: Optional[IqoalaVar],
        args: List[IqoalaVar],
        attr: Optional[IqoalaValue],
    ):
        if result is not None:
            if not isinstance(result, IqoalaTuple) and not isinstance(
                result, IqoalaVector
            ):
                raise HostLanguageSyntaxError(
                    f"{cls.OP_NAME} operation cannot have a result of type {type(result)}. "
                    f"It must be either IqoalaTuple or IqoalaVector."
                )
        if len(args) == 0:
            args = [IqoalaTuple([])]

        if len(args) != 1:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation takes 1 argument but got {len(args)}."
            )
        if not isinstance(args[0], IqoalaTuple):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation argument must be an IqoalaTuple."
            )
        if attr is None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation must have an attribute."
            )
        if not isinstance(attr, str):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation attribute must be a string."
            )

        return cls(result, args[0], attr)

    @property
    def req_routine(self) -> str:
        assert isinstance(self.attributes[0], str)
        return self.attributes[0]

    def __str__(self) -> str:
        return super().__str__()


class ReturnResultOp(ClassicalIqoalaOp):
    OP_NAME = "return_result"
    TYP = IqoalaInstructionType.CL

    def __init__(self, value: Union[IqoalaSingleton, IqoalaVector]) -> None:
        super().__init__(arguments=[value])

    @classmethod
    def from_generic_args(
        cls,
        result: Optional[IqoalaVar],
        args: List[IqoalaVar],
        attr: Optional[IqoalaValue],
    ):
        if result is not None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation cannot have a result."
            )
        if len(args) != 1:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation takes 1 argument but got {len(args)}."
            )
        if attr is not None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation cannot have an attribute."
            )
        if not isinstance(args[0], IqoalaSingleton) and not isinstance(
            args[0], IqoalaVector
        ):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation argument must be a string or vector."
            )

        return cls(args[0])


class JumpOp(ClassicalIqoalaOp):
    OP_NAME = "jump"
    TYP = IqoalaInstructionType.CL

    def __init__(self, block_name: str) -> None:
        super().__init__(attributes=[block_name])

    @classmethod
    def from_generic_args(
        cls,
        result: Optional[IqoalaVar],
        args: List[IqoalaVar],
        attr: Optional[IqoalaValue],
    ):
        if result is not None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation cannot have a result."
            )
        if len(args) != 0:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation takes 0 arguments but got {len(args)}."
            )
        if attr is None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation must have an attribute."
            )
        if not isinstance(attr, str):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation attribute must be a string."
            )
        return cls(attr)


class BranchIfEqualOp(ClassicalIqoalaOp):
    OP_NAME = "beq"
    TYP = IqoalaInstructionType.CL

    def __init__(
        self,
        value0: Union[IqoalaSingleton, IqoalaVectorElement],
        value1: Union[IqoalaSingleton, IqoalaVectorElement],
        block_name: str,
    ) -> None:
        super().__init__(arguments=[value0, value1], attributes=[block_name])

    @classmethod
    def from_generic_args(
        cls,
        result: Optional[IqoalaVar],
        args: List[IqoalaVar],
        attr: Optional[IqoalaValue],
    ):
        if result is not None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation cannot have a result."
            )
        if len(args) != 2:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation takes 2 arguments but got {len(args)}."
            )
        if attr is None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation must have an attribute."
            )
        if not isinstance(attr, str):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation attribute must be a string."
            )
        if (
            not isinstance(args[0], IqoalaSingleton)
            and not isinstance(args[0], IqoalaVectorElement)
        ) or (
            not isinstance(args[1], IqoalaSingleton)
            and not isinstance(args[1], IqoalaVectorElement)
        ):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation arguments must be strings or vector elements."
            )

        return cls(args[0], args[1], attr)


class BranchIfNotEqualOp(ClassicalIqoalaOp):
    OP_NAME = "bne"
    TYP = IqoalaInstructionType.CL

    def __init__(
        self,
        value0: Union[IqoalaSingleton, IqoalaVectorElement],
        value1: Union[IqoalaSingleton, IqoalaVectorElement],
        block_name: str,
    ) -> None:
        super().__init__(arguments=[value0, value1], attributes=[block_name])

    @classmethod
    def from_generic_args(
        cls,
        result: Optional[IqoalaVar],
        args: List[IqoalaVar],
        attr: Optional[IqoalaValue],
    ):
        if result is not None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation cannot have a result."
            )
        if len(args) != 2:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation takes 2 arguments but got {len(args)}."
            )
        if attr is None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation must have an attribute."
            )
        if not isinstance(attr, str):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation attribute must be a string."
            )
        if (
            not isinstance(args[0], IqoalaSingleton)
            and not isinstance(args[0], IqoalaVectorElement)
        ) or (
            not isinstance(args[1], IqoalaSingleton)
            and not isinstance(args[1], IqoalaVectorElement)
        ):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation arguments must be strings or vector elements."
            )
        return cls(args[0], args[1], attr)


class BranchIfGreaterThanOp(ClassicalIqoalaOp):
    OP_NAME = "bgt"
    TYP = IqoalaInstructionType.CL

    def __init__(
        self,
        value0: Union[IqoalaSingleton, IqoalaVectorElement],
        value1: Union[IqoalaSingleton, IqoalaVectorElement],
        block_name: str,
    ) -> None:
        super().__init__(arguments=[value0, value1], attributes=[block_name])

    @classmethod
    def from_generic_args(
        cls,
        result: Optional[IqoalaVar],
        args: List[IqoalaVar],
        attr: Optional[IqoalaValue],
    ):
        if result is not None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation cannot have a result."
            )
        if len(args) != 2:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation takes 2 arguments but got {len(args)}."
            )
        if attr is None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation must have an attribute."
            )
        if not isinstance(attr, str):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation attribute must be a string."
            )
        if (
            not isinstance(args[0], IqoalaSingleton)
            and not isinstance(args[0], IqoalaVectorElement)
        ) or (
            not isinstance(args[1], IqoalaSingleton)
            and not isinstance(args[1], IqoalaVectorElement)
        ):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation arguments must be strings or vector elements."
            )
        return cls(args[0], args[1], attr)


class BranchIfLessThanOp(ClassicalIqoalaOp):
    OP_NAME = "blt"
    TYP = IqoalaInstructionType.CL

    def __init__(
        self,
        value0: Union[IqoalaSingleton, IqoalaVectorElement],
        value1: Union[IqoalaSingleton, IqoalaVectorElement],
        block_name: str,
    ) -> None:
        super().__init__(arguments=[value0, value1], attributes=[block_name])

    @classmethod
    def from_generic_args(
        cls,
        result: Optional[IqoalaVar],
        args: List[IqoalaVar],
        attr: Optional[IqoalaValue],
    ):
        if result is not None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation cannot have a result."
            )
        if len(args) != 2:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation takes 2 arguments but got {len(args)}."
            )
        if attr is None:
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation must have an attribute."
            )
        if not isinstance(attr, str):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation attribute must be a string."
            )
        if (
            not isinstance(args[0], IqoalaSingleton)
            and not isinstance(args[0], IqoalaVectorElement)
        ) or (
            not isinstance(args[1], IqoalaSingleton)
            and not isinstance(args[1], IqoalaVectorElement)
        ):
            raise HostLanguageSyntaxError(
                f"{cls.OP_NAME} operation arguments must be strings or vector elements."
            )
        return cls(args[0], args[1], attr)


class BasicBlockType(Enum):
    CL = 0
    CC = auto()
    QL = auto()
    QC = auto()


@dataclass
class BasicBlock:
    name: str
    typ: BasicBlockType
    instructions: List[ClassicalIqoalaOp]
    deadlines: Optional[Dict[str, int]] = None

    def __str__(self) -> str:
        annotations = f"type = {self.typ.name}"
        if self.deadlines is not None:
            annotations += f", deadlines: {self.deadlines}"
        annotations = "{" + annotations + "}"
        s = f"^{self.name} {annotations}:\n"
        return s + "\n".join("    " + str(i) for i in self.instructions)
