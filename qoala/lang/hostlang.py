from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Union

from netqasm.lang.operand import Template

IqoalaValue = Union[int, Template, str]


class HostLanguageSyntaxError(Exception):
    """
    Exception raised when the host language code is syntactically incorrect.
    """

    pass


class IqoalaInstructionType(Enum):
    """
    Types of instructions in the Iqoala language.

    CC = Classical Communication
    CL = Classical Logic
    QC = Quantum Communication
    QL = Quantum Logic
    """

    CC = 0
    CL = auto()
    QC = auto()
    QL = auto()


class IqoalaVar:
    """
    Base class for all Iqoala variables.
    """

    pass


@dataclass(frozen=True)
class IqoalaSingleton(IqoalaVar):
    """
    A singleton variable in Iqoala. It stores a name of the variable. The value of this variable can be accessed
    by using this name in the host memory.

    :param name: Name of the variable.
    """

    name: str

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class IqoalaTuple(IqoalaVar):
    """
    A tuple variable in Iqoala. It stores a list of names of the variables. The values of these variables
    can be accessed by using these names in the host memory.

    :param values: List of names of the variables.
    """

    values: List[str]

    def __str__(self) -> str:
        return f"tuple<{';'.join(v for v in self.values)}>"


@dataclass(frozen=True)
class IqoalaVector(IqoalaVar):
    """
    A vector variable in Iqoala. It stores a name of the variable and a size of the vector. It basically represents an
    array of numbers with the given size. The values of this vector can be accessed by using the name of the variable
    in the host memory.

    :param name: Name of the variable.
    :param size: Size of the vector. It doesn't need to be stated explicitly as an integer. It can be a variable or a
    program input.
    """

    name: str
    size: IqoalaValue

    def __str__(self) -> str:
        return f"{self.name}<{self.size}>"


@dataclass(frozen=True)
class IqoalaVectorElement(IqoalaVar):
    """
    A vector element variable in Iqoala. It stores a name of the vector variable and an index of the element.

    :param name: Name of the vector variable.
    :param index: Index of the element in the vector.
    """

    name: str
    index: int

    def __str__(self) -> str:
        return f"{self.name}[{self.index}]"


class ClassicalIqoalaOp:
    """
    Base class for all classical Iqoala operations. Every operator has a type and a unique name. In general,
    instructions in Iqoala is written as follows:
    result = op_name(arg1, arg2, ..., argn) : attr1, attr2, ..., attrm

    :param arguments: List of arguments of the operation. It can be empty.
    :param results: Result of the operation. It can be None.
    :param attributes: List of attributes of the operation. It can be empty.
    """

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
        """
        Create an instance of the operation from the generic arguments. This method is used by the parser to create
        an instance of the operation from the parsed arguments. It is overridden by every subclass.
        """
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
    """
    A Classical Logic operation that assigns the constant integer to a singleton variable.

    Iqoala Example:
    x = assign_cval() : 3

    This example assigns the value 3 to the singleton variable with name x.
    """

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
            raise HostLanguageSyntaxError(
                "Result of assign_cval must be a singleton variable."
            )
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
    """
    A Classical Logic operation that waits for a period of time. The time(in ns) is given as an
    attribute of the operation.

    Iqoala Example:
    busy() : 500

    This example waits for 500 ns before executing the next operation.
    """

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
    """
    A Classical Communication operation that sends a message to another node. The message is sent through a classical
    socket. A variable storing the classical socket id that will be used to send the message will be given as the
    first argument of the operation. The variable storing the message will be given as the second argument of the
    operation. Argument variables can be singleton or vector element variables.

    Iqoala Example:
    send_cmsg(csocket, m)

    This example sends the value of variable m through a classical socket whose id is stored in variable csocket.
    """

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
    """
    A Classical Communication operation that receives a message from another node. The message is received through
    a classical socket. A variable storing the classical socket id that will be used to send the message will be given
    as an argument of the operation and the message will be stored in the variable given as the result of the operation.
    Argument can be a singleton or a vector element variable. Result must be a singleton variable.

    Iqoala Example:
    m = recv_cmsg(csocket)

    This example receives from a message from a classical socket whose id is stored in variable csocket and stores it in
    the singleton variable m.
    """

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
    """
    A Classical Logic operation that adds the values of the two variables and stores the result in
    a singleton variable. Argument variables can be singleton or vector element variables. Result must be a singleton
    variable.

    Iqoala Example:
    z = add_cval_c(x, y)

    This example adds the values of the variables x and y and stores the result in the variable z.
    """

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
    """
    A Classical Logic operation that multiplies the value of a variable with the given constant integer and stores
    the result in a singleton variable. Argument variable can be a singleton or a vector element variable. Result must
    be a singleton variable.

    Iqoala Example:
    y = mult_const(x) : 3

    This example multiplies the value of the variable x with 3 and stores the result in the variable y.
    """

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
    """
    A Classical Logic operation that takes two arguments and a one attribute. If the value of the second argument
    is 1, it multiplies the value of the first argument with the given constant and stores the result in a singleton
    variable. Otherwise, it stores the value of the first argument in the singleton variable. Argument variables
    can be singleton or vector element variables. Result must be a singleton variable.

    Iqoala Example:
    z = bcond_mult_const(x,y) : 3

    This example multiplies the value of the variable x with 3 and stores the result in the variable z if the value
    of the variable y is 1. Otherwise, it stores the value of the variable x in the variable z.
    """

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
    """
    A Classical Logic operation that calls the given subroutine. The subroutine name is given as an attribute of the
    operation. The arguments of the subroutine are given as the arguments of the operation. The result of the
    subroutine is stored in a variable given as the result of the operation. Arguments are given as a tuple
    variable. The result of the subroutine can be a tuple or a vector variable.

    Iqoala Example:
    tuple<m> = run_subroutine(tuple<x>) : subrt

    This example calls the subroutine with name subrt with the argument x and stores the result in the variable m.
    """

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
    """
    A Classical Logic operation that calls the given request routine. The request routine name is given as an
    attribute of the operation. The arguments of the request routine are given as the arguments of the operation.
    The result of the request routine is stored in a variable given as the result of the operation. Arguments are
    given as a tuple variable. The result of the subroutine can be a tuple or a vector variable.

    Iqoala Example:
    tuple<m> = run_request(tuple<x; y>) : req

    This example calls the request routine with name req with the arguments x and y and stores the result in the
    variable m.
    """

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
    """
    A Classical Logic operation that stores the value of the given variable in the result of the process. Argument
    variable can be a singleton or a vector variable.


    Iqoala Example:
    return_result(x)

    This example stores the value of the variable x in the result of the process.
    """

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
    """
    A Classical Logic operation that jumps to block with given block name. Block name is given as an attribute.

    Iqoala Example:
    jump() : blk2

    This example jumps to the block with name blk2.
    """

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
    """
    A Classical Logic operation that jumps to block with given block name if the values of the two variables are
    equal. Otherwise, it continues to the next operation. Block name is given as an attribute. Variables in the
    arguments can be either singleton variables or vector elements.

    Iqoala Example:
    beq(x, y) : blk2

    This example jumps to the block with name blk2 if the values of the variables x and y are equal.
    """

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
    """
    A Classical Logic operation that jumps to block with given block name if the values of the two variables are not
    equal. Otherwise, it continues to the next operation. Block name is given as an attribute. Variables in the
    arguments can be either singleton variables or vector elements.

    Iqoala Example:
    bne(x, y) : blk2

    This example jumps to the block with name blk2 if the values of the variables x and y are not equal.
    """

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
    """
    A Classical Logic operation that jumps to block with given block name if the value of the first variable is
    greater than the value of the second variable. Otherwise, it continues to the next operation. Block name is
     given as an attribute. Variables in the arguments can be either singleton variables or vector elements.

    Iqoala Example:
    bgt(x, y) : blk2

    This example jumps to the block with name blk2 if the value of the variable x is greater than the value of the
    variable y.
    """

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
    """
    A Classical Logic operation that jumps to block with given block name if the value of the first variable is
    less than the value of the second variable. Otherwise, it continues to the next operation. Block name is
     given as an attribute. Variables in the arguments can be either singleton variables or vector elements.

    Iqoala Example:
    blt(x, y) : blk2

    This example jumps to the block with name blk2 if the value of the variable x is less than the value of the
    variable y.
    """

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
    """
    Types of blocks in the Iqoala language.

    CC = Classical Communication
    CL = Classical Logic
    QC = Quantum Communication
    QL = Quantum Logic
    """

    CL = 0
    CC = auto()
    QL = auto()
    QC = auto()


@dataclass
class BasicBlock:
    """
    A basic block in the Iqoala language. A basic block is a sequence of instructions that are executed sequentially.
    Blocks only contain Classical Iqoala Operations.

    name: name of the block
    typ: type of the block
    instructions: list of instructions in the block
    deadlines: optional deadlines for the block. It is stored as a dictionary where the keys are the names of the
    variables and the values are the deadlines for the variables. if 'blk1 : 10' is in the deadlines dictionary, it
    means that the block must be executed within 10 seconds after the execution of the block with name blk1.
    """

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
