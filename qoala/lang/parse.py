from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from netqasm.lang.instr.flavour import Flavour, VanillaFlavour
from netqasm.lang.operand import Template
from netqasm.lang.parsing.text import parse_text_subroutine

from qoala.lang import hostlang as hl
from qoala.lang.program import LocalRoutine, ProgramMeta, QoalaProgram
from qoala.lang.request import (
    CallbackType,
    EprRole,
    EprType,
    QoalaRequest,
    RequestRoutine,
    RequestVirtIdMapping,
    RrReturnVector,
)
from qoala.lang.routine import LrReturnVector, RoutineMetadata

LHR_OP_NAMES: Dict[str, hl.ClassicalIqoalaOp] = {
    cls.OP_NAME: cls  # type: ignore
    for cls in [
        hl.SendCMsgOp,
        hl.ReceiveCMsgOp,
        hl.AddCValueOp,
        hl.MultiplyConstantCValueOp,
        hl.BitConditionalMultiplyConstantCValueOp,
        hl.AssignCValueOp,
        hl.RunSubroutineOp,
        hl.RunRequestOp,
        hl.ReturnResultOp,
        hl.BusyOp,
        hl.JumpOp,
        hl.BranchIfEqualOp,
        hl.BranchIfNotEqualOp,
        hl.BranchIfLessThanOp,
        hl.BranchIfGreaterThanOp,
    ]
}


class EndOfTextException(Exception):
    pass


class QoalaParseError(Exception):
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

    def _parse_meta_mapping(self, values: List[str]) -> Dict[int, str]:
        result_dict = {}
        for v in values:
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
                raise QoalaParseError("Could not parse meta.")
        except AssertionError:
            raise QoalaParseError
        except EndOfTextException:
            raise QoalaParseError

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

    def _parse_var(self, var_str: str) -> Union[str, hl.IqoalaTuple, hl.IqoalaVector]:
        if var_str.startswith("tuple<"):
            tup_values_str = var_str[6:-1]
            if len(tup_values_str) == 0:
                tup_values = []
            else:
                tup_values = [x.strip() for x in tup_values_str.split(";")]
            return hl.IqoalaTuple(tup_values)
        elif "<" in var_str:
            if not var_str.endswith(">"):
                raise QoalaParseError
            vec_split = var_str.split("<")
            vec_name = vec_split[0]
            vec_size_str = vec_split[1][:-1]  # strip last ">"
            vec_size: Union[str, int]
            try:
                vec_size = int(vec_size_str)
            except ValueError:
                vec_size = vec_size_str
            return hl.IqoalaVector(vec_name, vec_size)
        else:
            return var_str

    def _parse_lhr(self) -> hl.ClassicalIqoalaOp:
        line = self._read_line()

        attr: Optional[hl.IqoalaValue]

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

        lhr_op = LHR_OP_NAMES[op].from_generic_args(result, args, attr)  # type: ignore
        return lhr_op

    def parse(self) -> List[hl.ClassicalIqoalaOp]:
        instructions: List[hl.ClassicalIqoalaOp] = []

        try:
            while True:
                instr = self._parse_lhr()
                instructions.append(instr)
        except AssertionError:
            raise QoalaParseError
        except EndOfTextException:
            pass

        return instructions


class HostCodeParser:
    def __init__(self, text: str) -> None:
        self._text = text
        lines = [line.strip() for line in text.split("\n")]
        self._lines = [line for line in lines if len(line) > 0]
        self._lineno: int = 0

    def get_block_texts(self) -> List[str]:
        block_start_lines: List[int] = []

        for i, line in enumerate(self._lines):
            if line.startswith("^"):
                block_start_lines.append(i)

        assert len(block_start_lines) > 0

        block_texts: List[str] = []
        for i in range(len(block_start_lines) - 1):
            start = block_start_lines[i]
            end = block_start_lines[i + 1]
            text = self._lines[start:end]
            block_texts.append("\n".join([line for line in text]))

        last = block_start_lines[-1]
        last_text = self._lines[last:]
        block_texts.append("\n".join([line for line in last_text]))

        return block_texts

    def _parse_deadlines(self, text: str) -> Dict[str, int]:
        open_bracket = text.find("[")
        assert open_bracket >= 0
        close_bracket = text.find("]")
        assert close_bracket >= 0
        items = text[open_bracket + 1 : close_bracket].split(",")
        deadlines = {}
        for item in items:
            blk, dl = [i.strip() for i in item.split(":")]
            deadlines[blk] = int(dl)
        return deadlines

    def _parse_block_annotations(
        self, annotations: str
    ) -> Tuple[hl.BasicBlockType, Optional[Dict[str, int]]]:
        typ_index = annotations.find("type =")
        assert typ_index >= 0
        comma = annotations.find(",")
        if comma == -1:  # no deadline
            raw_typ = annotations[typ_index + 7 :]
            typ = hl.BasicBlockType[raw_typ.upper()]
            deadlines = None
        else:
            raw_typ = annotations[typ_index + 7 : comma]
            typ = hl.BasicBlockType[raw_typ.upper()]
            deadlines_str = annotations[comma + 1 :].strip()
            deadlines = self._parse_deadlines(deadlines_str[12:])
        return typ, deadlines

    def _parse_block_header(
        self, line: str
    ) -> Tuple[str, hl.BasicBlockType, Optional[Dict[str, int]]]:
        # return (block name, block type, block deadline)
        assert line.startswith("^")
        split_space = line.split(" ")
        assert len(split_space) > 0
        name = split_space[0][1:]  # trim '^' at start
        open_brace = line.find("{")
        assert open_brace >= 0
        close_brace = line.find("}")
        assert close_brace >= 0
        annotations_str = line[open_brace + 1 : close_brace]
        typ, deadline = self._parse_block_annotations(annotations_str)
        return name, typ, deadline

    def parse_block(self, text: str) -> hl.BasicBlock:
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if len(line) > 0]
        name, typ, deadline = self._parse_block_header(lines[0])
        instr_lines = lines[1:]
        instrs = IqoalaInstrParser("\n".join(instr_lines)).parse()

        return hl.BasicBlock(name, typ, instrs, deadline)

    def parse(self) -> List[hl.BasicBlock]:
        block_texts = self.get_block_texts()
        return [self.parse_block(text) for text in block_texts]


class LocalRoutineParser:
    def __init__(self, text: str, flavour: Optional[Flavour] = None) -> None:
        self._text = text
        lines = [line.strip() for line in text.split("\n")]
        self._lines = [line for line in lines if len(line) > 0]
        self._lineno: int = 0
        if flavour is None:
            flavour = VanillaFlavour()
        self._flavour = flavour

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

    def _parse_subrt_meta_line_with_vecs(
        self, key: str, line: str
    ) -> List[Union[str, LrReturnVector]]:
        split = line.split(":")
        assert len(split) >= 1
        assert split[0] == key
        if len(split) == 1:
            return []
        assert len(split) == 2
        if len(split[1]) == 0:
            return []
        values_str = split[1].split(",")
        values_str = [v.strip() for v in values_str]
        values: List[Union[str, LrReturnVector]] = []
        for v in values_str:
            if "<" in v:
                if not v.endswith(">"):
                    raise QoalaParseError
                vec_split = v.split("<")
                vec_name = vec_split[0]
                vec_size_str = vec_split[1][:-1]  # strip last ">"
                vec_size = int(vec_size_str)
                values.append(LrReturnVector(vec_name, vec_size))
            else:
                values.append(v)
        return values

    def _parse_subroutine(self) -> LocalRoutine:
        name_line = self._read_line()
        assert name_line.startswith("SUBROUTINE ")
        name = name_line[len("SUBROUTINE") + 1 :]
        params_line = self._parse_subrt_meta_line("params", self._read_line())
        # TODO: use params line?

        return_vars = self._parse_subrt_meta_line_with_vecs(
            "returns", self._read_line()
        )
        assert all(" " not in v for v in return_vars if isinstance(v, str))

        uses_line = self._parse_subrt_meta_line("uses", self._read_line())
        uses = [int(u) for u in uses_line]
        keeps_line = self._parse_subrt_meta_line("keeps", self._read_line())
        keeps = [int(k) for k in keeps_line]
        metadata = RoutineMetadata(qubit_use=uses, qubit_keep=keeps)

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

        subrt = parse_text_subroutine(subrt_text, flavour=self._flavour)

        # Check that all templates are declared as params to the subroutine
        if any(arg not in params_line for arg in subrt.arguments):
            raise QoalaParseError
        return LocalRoutine(name, subrt, return_vars, metadata, request_name)

    def parse(self) -> Dict[str, LocalRoutine]:
        subroutines: Dict[str, LocalRoutine] = {}
        try:
            while True:
                try:
                    subrt = self._parse_subroutine()
                    subroutines[subrt.name] = subrt
                except AssertionError:
                    raise QoalaParseError
        except EndOfTextException:
            return subroutines


class RequestRoutineParser:
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

    def _parse_request_line_with_vecs(
        self, key: str, line: str, allow_template: bool = False
    ) -> List[Union[str, RrReturnVector]]:
        split = line.split(":")
        assert len(split) >= 1
        assert split[0] == key
        if len(split) == 1:
            return []
        assert len(split) == 2
        if len(split[1]) == 0:
            return []
        values_str = split[1].split(",")
        values_str = [v.strip() for v in values_str]
        values: List[Union[str, RrReturnVector]] = []
        for v in values_str:
            if "<" in v:
                if not v.endswith(">"):
                    raise QoalaParseError
                vec_split = v.split("<")
                vec_name = vec_split[0]
                vec_size_str = vec_split[1][:-1]  # strip last ">"
                vec_size: Union[int, Template]
                if (
                    allow_template
                    and vec_size_str.startswith("{")
                    and vec_size_str.endswith("}")
                ):
                    vec_size_str = vec_size_str.strip("{}").strip()
                    vec_size = Template(vec_size_str)
                else:
                    vec_size = int(vec_size_str)
                values.append(RrReturnVector(vec_name, vec_size))
            else:
                values.append(v)
        return values

    def _parse_single_int_value(
        self, key: str, line: str, allow_template: bool = False
    ) -> Union[int, Template]:
        strings = self._parse_request_line(key, line)
        if len(strings) != 1:
            raise QoalaParseError
        value = strings[0]
        if allow_template:
            if value.startswith("{") and value.endswith("}"):
                value = value.strip("{}").strip()
                return Template(value)
        return int(value)

    def _parse_optional_str_value(self, key: str, line: str) -> Optional[str]:
        strings = self._parse_request_line(key, line)
        if len(strings) == 0:
            return None
        assert len(strings) == 1
        return strings[0]

    def _parse_int_list_value(self, key: str, line: str) -> List[int]:
        strings = self._parse_request_line(key, line)
        return [int(s) for s in strings]

    def _parse_single_float_value(
        self, key: str, line: str, allow_template: bool = False
    ) -> Union[float, Template]:
        strings = self._parse_request_line(key, line)
        if len(strings) != 1:
            raise QoalaParseError
        value = strings[0]
        if allow_template:
            if value.startswith("{") and value.endswith("}"):
                value = value.strip("{}").strip()
                return Template(value)
        return float(value)

    def _parse_epr_create_role_value(self, key: str, line: str) -> EprRole:
        strings = self._parse_request_line(key, line)
        if len(strings) != 1:
            raise QoalaParseError
        try:
            return EprRole[strings[0].upper()]
        except KeyError:
            raise QoalaParseError

    def _parse_epr_create_type_value(self, key: str, line: str) -> EprType:
        strings = self._parse_request_line(key, line)
        if len(strings) != 1:
            raise QoalaParseError
        try:
            return EprType[strings[0].upper()]
        except KeyError:
            raise QoalaParseError

    def _parse_virt_ids(self, key: str, line: str) -> RequestVirtIdMapping:
        split = line.split(":")
        assert len(split) >= 1
        assert split[0] == key
        value = split[1].strip()
        return RequestVirtIdMapping.from_str(value)

    def _parse_request(self) -> RequestRoutine:
        name_line = self._read_line()
        if not name_line.startswith("REQUEST "):
            raise QoalaParseError
        name = name_line[len("REQUEST") + 1 :]

        raw_callback_type = self._parse_request_line("callback_type", self._read_line())
        assert len(raw_callback_type) == 1
        callback_type = CallbackType[raw_callback_type[0].upper()]

        callback = self._parse_optional_str_value("callback", self._read_line())

        return_vars = self._parse_request_line_with_vecs(
            "return_vars", self._read_line(), allow_template=True
        )
        assert all(" " not in v for v in return_vars if isinstance(v, str))

        remote_id = self._parse_single_int_value(
            "remote_id", self._read_line(), allow_template=True
        )
        epr_socket_id = self._parse_single_int_value(
            "epr_socket_id", self._read_line(), allow_template=True
        )
        num_pairs = self._parse_single_int_value(
            "num_pairs", self._read_line(), allow_template=True
        )
        # virt_ids = self._parse_int_list_value("virt_ids", self._read_line())
        virt_ids = self._parse_virt_ids("virt_ids", self._read_line())
        timeout = self._parse_single_int_value(
            "timeout", self._read_line(), allow_template=True
        )
        fidelity = self._parse_single_float_value(
            "fidelity", self._read_line(), allow_template=True
        )
        typ = self._parse_epr_create_type_value("typ", self._read_line())
        role = self._parse_epr_create_role_value("role", self._read_line())

        request = QoalaRequest(
            name=name,
            remote_id=remote_id,
            epr_socket_id=epr_socket_id,
            num_pairs=num_pairs,
            virt_ids=virt_ids,
            timeout=timeout,
            fidelity=fidelity,
            typ=typ,
            role=role,
        )
        return RequestRoutine(name, request, return_vars, callback_type, callback)

    def parse(self) -> Dict[str, RequestRoutine]:
        requests: Dict[str, RequestRoutine] = {}
        try:
            while True:
                request = self._parse_request()
                requests[request.name] = request
        except EndOfTextException:
            return requests


class QoalaParser:
    def __init__(
        self,
        text: Optional[str] = None,
        meta_text: Optional[str] = None,
        host_text: Optional[str] = None,
        subrt_text: Optional[str] = None,
        req_text: Optional[str] = None,
        flavour: Optional[Flavour] = None,
    ) -> None:
        if text is not None:
            meta_text, host_text, subrt_text, req_text = self._split_text(text)
        else:
            assert meta_text is not None
            assert host_text is not None
            assert subrt_text is not None
            assert req_text is not None
        self._meta_text = meta_text
        self._host_text = host_text
        self._subrt_text = subrt_text
        self._req_text = req_text
        self._meta_parser = IqoalaMetaParser(meta_text)
        self._host_parser = HostCodeParser(host_text)
        self._subrt_parser = LocalRoutineParser(subrt_text, flavour)
        self._req_parser = RequestRoutineParser(req_text)

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
        host_end_line: Optional[int] = None
        if first_subrt_line is None and first_req_line is None:
            # no subroutines and no requests
            subrt_text = ""
            req_text = ""
        elif first_subrt_line is not None and first_req_line is None:
            # subroutines but no requests
            subrt_text = "\n".join(lines[first_subrt_line:])
            req_text = ""
            host_end_line = first_subrt_line
        elif first_subrt_line is None and first_req_line is not None:
            # no subroutines but only requests
            subrt_text = ""
            req_text = "\n".join(lines[first_req_line:])
            host_end_line = first_req_line
        else:
            # subroutines and requests
            subrt_text = "\n".join(lines[first_subrt_line:first_req_line])
            req_text = "\n".join(lines[first_req_line:])
            host_end_line = first_subrt_line
        if host_end_line is not None:
            host_text = "\n".join(lines[meta_end_line + 1 : host_end_line])
        else:
            host_text = "\n".join(lines[meta_end_line + 1 :])

        return meta_text, host_text, subrt_text, req_text

    def parse(self) -> QoalaProgram:
        blocks = self._host_parser.parse()
        subroutines = self._subrt_parser.parse()
        requests = self._req_parser.parse()
        meta = self._meta_parser.parse()

        # Check that all references to subroutines (in RunSubroutineOp instructions)
        # and to requests (in RunRequestOp instructions) are valid.
        for block in blocks:
            for instr in block.instructions:
                if isinstance(instr, hl.RunSubroutineOp):
                    subrt_name = instr.subroutine
                    if subrt_name not in subroutines:
                        raise QoalaParseError
                elif isinstance(instr, hl.RunRequestOp):
                    req_name = instr.req_routine
                    if req_name not in requests:
                        raise QoalaParseError

        return QoalaProgram(meta, blocks, subroutines, requests)
