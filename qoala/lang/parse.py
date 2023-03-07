from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from netqasm.lang.instr.flavour import NVFlavour
from netqasm.lang.operand import Template
from netqasm.lang.parsing.text import parse_text_subroutine

from qoala.lang.hostlang import (
    AddCValueOp,
    AssignCValueOp,
    BitConditionalMultiplyConstantCValueOp,
    ClassicalIqoalaOp,
    IqoalaSharedMemLoc,
    IqoalaValue,
    IqoalaVector,
    MultiplyConstantCValueOp,
    ReceiveCMsgOp,
    ReturnResultOp,
    RunSubroutineOp,
    SendCMsgOp,
)
from qoala.lang.program import IqoalaProgram, LocalRoutine, ProgramMeta
from qoala.lang.request import IqoalaRequest
from qoala.sim.requests import (
    EprCreateRole,
    EprCreateType,
    NetstackCreateRequest,
    NetstackReceiveRequest,
)

LHR_OP_NAMES: Dict[str, ClassicalIqoalaOp] = {
    cls.OP_NAME: cls  # type: ignore
    for cls in [
        SendCMsgOp,
        ReceiveCMsgOp,
        AddCValueOp,
        MultiplyConstantCValueOp,
        BitConditionalMultiplyConstantCValueOp,
        AssignCValueOp,
        RunSubroutineOp,
        ReturnResultOp,
    ]
}


class EndOfTextException(Exception):
    pass


class IqoalaParseError(Exception):
    pass


class IqoalaMetaParser:
    def __init__(self, text: str) -> None:
        self._text = text
        lines = [line.strip() for line in text.split("\n")]
        self._lines = [line for line in lines if len(line) > 0]
        self._lineno: int = 0

    def _next_line(self) -> None:
        self._lineno += 1

    def _read_line(self) -> str:
        while True:
            if self._lineno >= len(self._lines):
                raise EndOfTextException
            line = self._lines[self._lineno]
            self._next_line()
            if len(line) > 0:
                return line
            # if no non-empty line, will always break on EndOfLineException

    def _parse_meta_line(self, key: str, line: str) -> List[str]:
        split = line.split(":")
        assert len(split) >= 1
        assert split[0] == key
        if len(split) == 1:
            return []
        assert len(split) == 2
        if len(split[1]) == 0:
            return []
        values = split[1].split(",")
        return [v.strip() for v in values]

    def _parse_meta_mapping(self, value_str: str) -> Dict[int, str]:
        result_dict = {}
        for v in value_str:
            key_value = [x.strip() for x in v.split("->")]
            assert len(key_value) == 2
            result_dict[int(key_value[0].strip())] = key_value[1].strip()
        return result_dict

    def parse(self) -> ProgramMeta:
        try:
            start_line = self._read_line()
            assert start_line == "META_START"

            name_values = self._parse_meta_line("name", self._read_line())
            assert len(name_values) == 1
            name = name_values[0]

            parameters = self._parse_meta_line("parameters", self._read_line())

            csockets_map = self._parse_meta_line("csockets", self._read_line())
            csockets = self._parse_meta_mapping(csockets_map)
            epr_sockets_map = self._parse_meta_line("epr_sockets", self._read_line())
            epr_sockets = self._parse_meta_mapping(epr_sockets_map)

            end_line = self._read_line()
            if end_line != "META_END":
                raise IqoalaParseError("Could not parse meta.")
        except AssertionError:
            raise IqoalaParseError
        except EndOfTextException:
            raise IqoalaParseError

        return ProgramMeta(name, parameters, csockets, epr_sockets)


class IqoalaInstrParser:
    def __init__(self, text: str) -> None:
        self._text = text
        lines = [line.strip() for line in text.split("\n")]
        self._lines = [line for line in lines if len(line) > 0]
        self._lineno: int = 0

    def _next_line(self) -> None:
        self._lineno += 1

    def _read_line(self) -> str:
        while True:
            if self._lineno >= len(self._lines):
                raise EndOfTextException
            line = self._lines[self._lineno]
            self._next_line()
            if len(line) > 0:
                return line
            # if no non-empty line, will always break on EndOfLineException

    def _parse_var(self, var_str: str) -> Union[str, IqoalaVector]:
        if var_str.startswith("vec<"):
            vec_values_str = var_str[4:-1]
            if len(vec_values_str) == 0:
                vec_values = []
            else:
                vec_values = [x.strip() for x in vec_values_str.split(";")]
            return IqoalaVector(vec_values)
        else:
            return var_str

    def _parse_lhr(self) -> ClassicalIqoalaOp:
        line = self._read_line()

        attr: Optional[IqoalaValue]

        assign_parts = [x.strip() for x in line.split("=")]
        assert len(assign_parts) <= 2
        if len(assign_parts) == 1:
            value = assign_parts[0]
            result = None
        elif len(assign_parts) == 2:
            value = assign_parts[1]
            result = self._parse_var(assign_parts[0])
        value_parts = [x.strip() for x in value.split(":")]
        assert len(value_parts) <= 2
        if len(value_parts) == 2:
            value = value_parts[0]
            attr_str = value_parts[1]
            try:
                attr = int(attr_str)
            except ValueError:
                attr = attr_str
        else:
            value = value_parts[0]
            attr = None

        op_parts = [x.strip() for x in value.split("(")]
        assert len(op_parts) == 2
        op = op_parts[0]
        arguments = op_parts[1].rstrip(")")
        if len(arguments) == 0:
            raw_args = []
        else:
            raw_args = [x.strip() for x in arguments.split(",")]

        args = [self._parse_var(arg) for arg in raw_args]

        # print(f"result = {result}, op = {op}, args = {args}, attr = {attr}")

        lhr_op = LHR_OP_NAMES[op].from_generic_args(result, args, attr)
        return lhr_op

    def parse(self) -> List[ClassicalIqoalaOp]:
        instructions: List[ClassicalIqoalaOp] = []

        try:
            while True:
                instr = self._parse_lhr()
                instructions.append(instr)
        except AssertionError:
            raise IqoalaParseError
        except EndOfTextException:
            pass

        return instructions


class LocalRoutineParser:
    def __init__(self, text: str) -> None:
        self._text = text
        lines = [line.strip() for line in text.split("\n")]
        self._lines = [line for line in lines if len(line) > 0]
        self._lineno: int = 0

    def _next_line(self) -> None:
        self._lineno += 1

    def _read_line(self) -> str:
        while True:
            if self._lineno >= len(self._lines):
                raise EndOfTextException
            line = self._lines[self._lineno]
            self._next_line()
            if len(line) > 0:
                return line
            # if no non-empty line, will always break on EndOfLineException

    def _parse_subrt_meta_line(self, key: str, line: str) -> List[str]:
        split = line.split(":")
        assert len(split) >= 1
        assert split[0] == key
        if len(split) == 1:
            return []
        assert len(split) == 2
        if len(split[1]) == 0:
            return []
        values = split[1].split(",")
        return [v.strip() for v in values]

    def _parse_nqasm_return_mapping(
        self, value_str: str
    ) -> Dict[str, IqoalaSharedMemLoc]:
        result_dict = {}
        for v in value_str:
            key_value = [x.strip() for x in v.split("->")]
            assert len(key_value) == 2
            result_dict[key_value[1]] = IqoalaSharedMemLoc(key_value[0])
        return result_dict

    def _parse_subroutine(self) -> LocalRoutine:
        return_map: Dict[str, IqoalaSharedMemLoc] = {}
        name_line = self._read_line()
        assert name_line.startswith("SUBROUTINE ")
        name = name_line[len("SUBROUTINE") + 1 :]
        params_line = self._parse_subrt_meta_line("params", self._read_line())
        # TODO: use params line?
        return_map_line = self._parse_subrt_meta_line("returns", self._read_line())
        return_map = self._parse_nqasm_return_mapping(return_map_line)
        request_line = self._parse_subrt_meta_line("request", self._read_line())
        assert len(request_line) in [0, 1]
        request_name = None if len(request_line) == 0 else request_line[0]

        start_line = self._read_line()
        assert start_line == "NETQASM_START"
        subrt_lines = []
        while True:
            line = self._read_line()
            if line == "NETQASM_END":
                break
            subrt_lines.append(line)
        subrt_text = "\n".join(subrt_lines)
        try:
            subrt = parse_text_subroutine(subrt_text)
        except KeyError:
            subrt = parse_text_subroutine(subrt_text, flavour=NVFlavour())

        # Check that all templates are declared as params to the subroutine
        if any(arg not in params_line for arg in subrt.arguments):
            raise IqoalaParseError
        return LocalRoutine(name, subrt, return_map, request_name)

    def parse(self) -> Dict[str, LocalRoutine]:
        subroutines: Dict[str, LocalRoutine] = {}
        try:
            while True:
                subrt = self._parse_subroutine()
                subroutines[subrt.name] = subrt
        except EndOfTextException:
            return subroutines


class IQoalaRequestParser:
    def __init__(self, text: str) -> None:
        self._text = text
        lines = [line.strip() for line in text.split("\n")]
        self._lines = [line for line in lines if len(line) > 0]
        self._lineno: int = 0

    def _next_line(self) -> None:
        self._lineno += 1

    def _read_line(self) -> str:
        while True:
            if self._lineno >= len(self._lines):
                raise EndOfTextException
            line = self._lines[self._lineno]
            self._next_line()
            if len(line) > 0:
                return line
            # if no non-empty line, will always break on EndOfLineException

    def _parse_request_line(self, key: str, line: str) -> List[str]:
        split = line.split(":")
        assert len(split) >= 1
        assert split[0] == key
        if len(split) == 1:
            return []
        assert len(split) == 2
        if len(split[1]) == 0:
            return []
        values = split[1].split(",")
        return [v.strip() for v in values]

    def _parse_single_int_value(
        self, key: str, line: str, allow_template: bool = False
    ) -> Union[int, Template]:
        strings = self._parse_request_line(key, line)
        if len(strings) != 1:
            raise IqoalaParseError
        value = strings[0]
        if allow_template:
            if value.startswith("{") and value.endswith("}"):
                value = value.strip("{}").strip()
                return Template(value)
        return int(value)

    def _parse_int_list_value(self, key: str, line: str) -> int:
        strings = self._parse_request_line(key, line)
        return [int(s) for s in strings]

    def _parse_single_float_value(self, key: str, line: str) -> int:
        strings = self._parse_request_line(key, line)
        if len(strings) != 1:
            raise IqoalaParseError
        return float(strings[0])

    def _parse_epr_create_role_value(self, key: str, line: str) -> int:
        strings = self._parse_request_line(key, line)
        if len(strings) != 1:
            raise IqoalaParseError
        try:
            return EprCreateRole[strings[0].upper()]
        except KeyError:
            raise IqoalaParseError

    def _parse_epr_create_type_value(self, key: str, line: str) -> int:
        strings = self._parse_request_line(key, line)
        if len(strings) != 1:
            raise IqoalaParseError
        try:
            return EprCreateType[strings[0].upper()]
        except KeyError:
            raise IqoalaParseError

    def _parse_request(self) -> IqoalaRequest:
        name_line = self._read_line()
        if not name_line.startswith("REQUEST "):
            raise IqoalaParseError
        name = name_line[len("REQUEST") + 1 :]

        role = self._parse_epr_create_role_value("role", self._read_line())
        remote_id = self._parse_single_int_value(
            "remote_id", self._read_line(), allow_template=True
        )
        epr_socket_id = self._parse_single_int_value("epr_socket_id", self._read_line())
        typ = self._parse_epr_create_type_value("typ", self._read_line())
        num_pairs = self._parse_single_int_value("num_pairs", self._read_line())
        fidelity = self._parse_single_float_value("fidelity", self._read_line())
        virt_qubit_ids = self._parse_int_list_value("virt_qubit_ids", self._read_line())
        result_array_addr = self._parse_single_int_value(
            "result_array_addr", self._read_line()
        )

        if role == EprCreateRole.CREATE:
            request = NetstackCreateRequest(
                remote_id=remote_id,
                epr_socket_id=epr_socket_id,
                typ=typ,
                num_pairs=num_pairs,
                fidelity=fidelity,
                virt_qubit_ids=virt_qubit_ids,
                result_array_addr=result_array_addr,
            )
        else:
            assert role == EprCreateRole.RECEIVE
            request = NetstackReceiveRequest(
                remote_id=remote_id,
                epr_socket_id=epr_socket_id,
                typ=typ,
                num_pairs=num_pairs,
                fidelity=fidelity,
                virt_qubit_ids=virt_qubit_ids,
                result_array_addr=result_array_addr,
            )

        return IqoalaRequest(name=name, role=role, request=request)

    def parse(self) -> Dict[str, IqoalaRequest]:
        requests: Dict[str, IqoalaRequest] = {}
        try:
            while True:
                request = self._parse_request()
                requests[request.name] = request
        except EndOfTextException:
            return requests


class IqoalaParser:
    def __init__(
        self,
        text: Optional[str] = None,
        meta_text: Optional[str] = None,
        instr_text: Optional[str] = None,
        subrt_text: Optional[str] = None,
        req_text: Optional[str] = None,
    ) -> None:
        if text is not None:
            meta_text, instr_text, subrt_text, req_text = self._split_text(text)
        else:
            assert meta_text is not None
            assert instr_text is not None
            assert subrt_text is not None
            assert req_text is not None
        self._meta_text = meta_text
        self._instr_text = instr_text
        self._subrt_text = subrt_text
        self._req_text = req_text
        self._meta_parser = IqoalaMetaParser(meta_text)
        self._instr_parser = IqoalaInstrParser(instr_text)
        self._subrt_parser = LocalRoutineParser(subrt_text)
        self._req_parser = IQoalaRequestParser(req_text)

    def _split_text(self, text: str) -> Tuple[str, str, str, str]:
        lines = [line.strip() for line in text.split("\n")]
        meta_end_line: int
        first_subrt_line: Optional[int] = None
        first_req_line: Optional[int] = None
        for i, line in enumerate(lines):
            if "META_END" in line:
                meta_end_line = i
                break
        for i, line in enumerate(lines):
            if "SUBROUTINE" in line:
                first_subrt_line = i
                break
        for i, line in enumerate(lines):
            if "REQUEST" in line:
                first_req_line = i
                break

        meta_text = "\n".join(lines[0 : meta_end_line + 1])
        instr_text = "\n".join(lines[meta_end_line + 1 : first_subrt_line])
        if first_subrt_line is None:
            # no subroutines and no requests
            subrt_text = ""
            req_text = ""
        elif first_req_line is None:
            # subroutines but no requests
            subrt_text = "\n".join(lines[first_subrt_line:])
            req_text = ""
        else:
            # subroutines and requests
            subrt_text = "\n".join(lines[first_subrt_line:first_req_line])
            req_text = "\n".join(lines[first_req_line:])

        return meta_text, instr_text, subrt_text, req_text

    def parse(self) -> IqoalaProgram:
        instructions = self._instr_parser.parse()
        subroutines = self._subrt_parser.parse()
        requests = self._req_parser.parse()
        meta = self._meta_parser.parse()

        # Check that all references to subroutines (in RunSubroutineOp instructions)
        # are valid.
        for instr in instructions:
            if isinstance(instr, RunSubroutineOp):
                subrt_name = instr.subroutine
                if subrt_name not in subroutines:
                    raise IqoalaParseError
        return IqoalaProgram(instructions, subroutines, meta, requests)
