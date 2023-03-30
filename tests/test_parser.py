import os

import pytest
from netqasm.lang.instr.core import MeasInstruction, SetInstruction
from netqasm.lang.operand import Register, Template
from netqasm.lang.subroutine import Subroutine

from qoala.lang.hostlang import (
    AssignCValueOp,
    BasicBlockType,
    IqoalaVector,
    RunSubroutineOp,
)
from qoala.lang.parse import (
    HostCodeParser,
    IqoalaInstrParser,
    IqoalaMetaParser,
    IqoalaParseError,
    IqoalaParser,
    LocalRoutineParser,
    RequestRoutineParser,
)
from qoala.lang.program import LocalRoutine, ProgramMeta
from qoala.lang.request import (
    CallbackType,
    EprRole,
    EprType,
    IqoalaRequest,
    RequestRoutine,
    RequestVirtIdMapping,
)
from qoala.lang.routine import RoutineMetadata
from qoala.util.tests import text_equal


def test_parse_incomplete_meta():
    text = """
META_START
name: alice
parameters: 
csockets: 0 -> bob
META_END
    """

    with pytest.raises(IqoalaParseError):
        IqoalaMetaParser(text).parse()


def test_parse_meta_no_end():
    text = """
META_START
name: alice
parameters: 
csockets: 0 -> bob
epr_sockets: 
    """

    with pytest.raises(IqoalaParseError):
        IqoalaMetaParser(text).parse()


def test_parse_meta():
    text = """
META_START
name: alice
parameters: 
csockets: 0 -> bob
epr_sockets: 
META_END
    """

    meta = IqoalaMetaParser(text).parse()

    assert meta == ProgramMeta(
        name="alice", parameters=[], csockets={0: "bob"}, epr_sockets={}
    )


def test_parse_meta_multiple_remotes():
    text = """
META_START
name: alice
parameters: theta1, theta2
csockets: 0 -> bob, 1 -> charlie
epr_sockets: 0 -> bob
META_END
    """

    meta = IqoalaMetaParser(text).parse()

    assert meta == ProgramMeta(
        name="alice",
        parameters=["theta1", "theta2"],
        csockets={0: "bob", 1: "charlie"},
        epr_sockets={0: "bob"},
    )


def test_parse_1_instr():
    text = """
x = assign_cval() : 1
    """

    instructions = IqoalaInstrParser(text).parse()

    assert len(instructions) == 1
    assert instructions[0] == AssignCValueOp(result="x", value=1)


def test_parse_2_instr():
    text = """
x = assign_cval() : 1
y = assign_cval() : 17
    """

    instructions = IqoalaInstrParser(text).parse()

    assert len(instructions) == 2
    assert instructions[0] == AssignCValueOp(result="x", value=1)
    assert instructions[1] == AssignCValueOp(result="y", value=17)


def test_parse_faulty_instr():
    text = """
x = assign_cval
    """

    with pytest.raises(IqoalaParseError):
        IqoalaInstrParser(text).parse()


def test_parse_vec():
    text = """
run_subroutine(vec<x>) : subrt1
    """

    instructions = IqoalaInstrParser(text).parse()
    assert len(instructions) == 1
    assert instructions[0] == RunSubroutineOp(
        result=None, values=IqoalaVector(["x"]), subrt="subrt1"
    )


def test_parse_vec_2_elements():
    text = """
run_subroutine(vec<x; y>) : subrt1
    """

    instructions = IqoalaInstrParser(text).parse()
    assert len(instructions) == 1
    assert instructions[0] == RunSubroutineOp(
        result=None, values=IqoalaVector(["x", "y"]), subrt="subrt1"
    )


def test_parse_vec_2_elements_and_return():
    text = """
vec<m> = run_subroutine(vec<x; y>) : subrt1
    """

    instructions = IqoalaInstrParser(text).parse()
    assert len(instructions) == 1
    assert instructions[0] == RunSubroutineOp(
        result=IqoalaVector(["m"]), values=IqoalaVector(["x", "y"]), subrt="subrt1"
    )


def test_parse_vec_2_elements_and_return_2_elements():
    text = """
vec<m1; m2> = run_subroutine(vec<x; y>) : subrt1
    """

    instructions = IqoalaInstrParser(text).parse()
    assert len(instructions) == 1
    assert instructions[0] == RunSubroutineOp(
        result=IqoalaVector(["m1", "m2"]),
        values=IqoalaVector(["x", "y"]),
        subrt="subrt1",
    )


def test_parse_block_header():
    text = "^b0 {type = CL}:"

    name, typ = HostCodeParser("")._parse_block_header(text)
    assert name == "b0"
    assert typ == BasicBlockType.CL


def test_parse_block():
    text = """
^b0 {type = CL}:
    x = assign_cval() : 1
    y = assign_cval() : 17
    """

    block = HostCodeParser("").parse_block(text)
    assert block.name == "b0"
    assert block.typ == BasicBlockType.CL
    assert len(block.instructions) == 2
    assert block.instructions[0] == AssignCValueOp(result="x", value=1)
    assert block.instructions[1] == AssignCValueOp(result="y", value=17)


def test_get_block_texts():
    text = """
^b0 {type = CL}:
    x = assign_cval() : 1
    y = assign_cval() : 17

^b1 {type = QL}:
    run_subroutine(vec<x>) : subrt1
    """

    block_texts = HostCodeParser(text).get_block_texts()
    assert len(block_texts) == 2


def test_parse_multiple_blocks():
    text = """
^b0 {type = CL}:
    x = assign_cval() : 1
    y = assign_cval() : 17

^b1 {type = QL}:
    run_subroutine(vec<x>) : subrt1
    """

    blocks = HostCodeParser(text).parse()
    assert len(blocks) == 2

    assert blocks[0].name == "b0"
    assert blocks[0].typ == BasicBlockType.CL
    assert len(blocks[0].instructions) == 2
    assert blocks[0].instructions[0] == AssignCValueOp(result="x", value=1)
    assert blocks[0].instructions[1] == AssignCValueOp(result="y", value=17)

    assert blocks[1].name == "b1"
    assert blocks[1].typ == BasicBlockType.QL
    assert len(blocks[1].instructions) == 1
    assert blocks[1].instructions[0] == RunSubroutineOp(
        result=None, values=IqoalaVector(["x"]), subrt="subrt1"
    )


def test_parse_subrt():
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
    assert len(parsed) == 1
    assert "subrt1" in parsed
    subrt = parsed["subrt1"]

    expected_instrs = [
        SetInstruction(reg=Register.from_str("Q0"), imm=Template("my_value"))
    ]
    expected_args = ["my_value"]
    assert subrt == LocalRoutine(
        name="subrt1",
        subrt=Subroutine(instructions=expected_instrs, arguments=expected_args),
        metadata=RoutineMetadata.use_none(),
        return_vars=["m"],
    )


def test_parse_subrt_2():
    text = """
SUBROUTINE my_subroutine
    params: param1, param2
    returns: result1, result2
    uses: 0, 1
    keeps: 
    request: 
  NETQASM_START
    set R0 {param1}
    set R1 {param2}
    meas Q0 M5
    meas Q1 M6
  NETQASM_END
    """

    parsed = LocalRoutineParser(text).parse()
    assert len(parsed) == 1
    assert "my_subroutine" in parsed
    subrt = parsed["my_subroutine"]

    expected_instrs = [
        SetInstruction(reg=Register.from_str("R0"), imm=Template("param1")),
        SetInstruction(reg=Register.from_str("R1"), imm=Template("param2")),
        MeasInstruction(reg0=Register.from_str("Q0"), reg1=Register.from_str("M5")),
        MeasInstruction(reg0=Register.from_str("Q1"), reg1=Register.from_str("M6")),
    ]
    expected_args = ["param1", "param2"]
    assert subrt == LocalRoutine(
        name="my_subroutine",
        subrt=Subroutine(instructions=expected_instrs, arguments=expected_args),
        metadata=RoutineMetadata.free_all([0, 1]),
        return_vars=["result1", "result2"],
    )


def test_parse_multiple_subrt():
    text = """
SUBROUTINE subrt1
    params: param1
    returns: m
    uses: 0
    keeps:
    request: 
  NETQASM_START
    set R0 {param1}
    meas Q0 M0
  NETQASM_END

SUBROUTINE subrt2
    params: theta
    returns: 
    uses: 
    keeps:
    request: 
  NETQASM_START
    set R0 {theta}
  NETQASM_END
    """

    parsed = LocalRoutineParser(text).parse()
    assert len(parsed) == 2
    assert "subrt1" in parsed
    assert "subrt2" in parsed
    subrt1 = parsed["subrt1"]
    subrt2 = parsed["subrt2"]

    expected_instrs_1 = [
        SetInstruction(reg=Register.from_str("R0"), imm=Template("param1")),
        MeasInstruction(reg0=Register.from_str("Q0"), reg1=Register.from_str("M0")),
    ]
    expected_args_1 = ["param1"]
    expected_instrs_2 = [
        SetInstruction(reg=Register.from_str("R0"), imm=Template("theta")),
    ]
    expected_args_2 = ["theta"]

    assert subrt1 == LocalRoutine(
        name="subrt1",
        subrt=Subroutine(instructions=expected_instrs_1, arguments=expected_args_1),
        metadata=RoutineMetadata.free_all([0]),
        return_vars=["m"],
    )
    assert subrt2 == LocalRoutine(
        name="subrt2",
        subrt=Subroutine(instructions=expected_instrs_2, arguments=expected_args_2),
        metadata=RoutineMetadata.use_none(),
        return_vars=[],
    )


def test_parse_invalid_subrt():
    text = """
SUBROUTINE my_subroutine
    params: param1, param2
    returns: result1, result2
    request: 
  NETQASM_START
    set R0 {param3}
  NETQASM_END
    """

    with pytest.raises(IqoalaParseError):
        LocalRoutineParser(text).parse()


def test_parse_request():
    text = """
REQUEST req1
  callback_type: wait_all
  callback: 
  return_vars: 
  remote_id: 1
  epr_socket_id: 0
  num_pairs: 5
  virt_ids: all 0
  timeout: 1000
  fidelity: 0.65
  typ: create_keep
  role: create
  result_array_addr: 3
    """

    parsed = RequestRoutineParser(text).parse()
    assert len(parsed) == 1
    assert "req1" in parsed
    routine = parsed["req1"]

    assert routine == RequestRoutine(
        name="req1",
        return_vars=[],
        callback_type=CallbackType.WAIT_ALL,
        callback=None,
        request=IqoalaRequest(
            name="req1",
            remote_id=1,
            epr_socket_id=0,
            num_pairs=5,
            virt_ids=RequestVirtIdMapping.from_str("all 0"),
            timeout=1000,
            fidelity=0.65,
            typ=EprType.CREATE_KEEP,
            role=EprRole.CREATE,
            result_array_addr=3,
        ),
    )


def test_parse_request_2():
    text = """
REQUEST req1
  callback_type: sequential
  callback: subrt1
  return_vars: 
  remote_id: 1
  epr_socket_id: 0
  num_pairs: 3
  virt_ids: custom 1, 2, 3
  timeout: 1000
  fidelity: 0.65
  typ: measure_directly
  role: receive
  result_array_addr: 0
    """

    parsed = RequestRoutineParser(text).parse()
    assert len(parsed) == 1
    assert "req1" in parsed
    routine = parsed["req1"]

    assert routine == RequestRoutine(
        name="req1",
        return_vars=[],
        callback_type=CallbackType.SEQUENTIAL,
        callback="subrt1",
        request=IqoalaRequest(
            name="req1",
            remote_id=1,
            epr_socket_id=0,
            num_pairs=3,
            virt_ids=RequestVirtIdMapping.from_str("custom 1, 2, 3"),
            timeout=1000,
            fidelity=0.65,
            typ=EprType.MEASURE_DIRECTLY,
            role=EprRole.RECEIVE,
            result_array_addr=0,
        ),
    )


def test_parse_request_with_template():
    text = """
REQUEST req1
  callback_type: wait_all
  callback: 
  return_vars: 
  remote_id: {client_id}
  epr_socket_id: 0
  num_pairs: 3
  virt_ids: custom 1, 2, 3
  timeout: 1000
  fidelity: 0.65
  typ: measure_directly
  role: receive
  result_array_addr: 0
    """

    parsed = RequestRoutineParser(text).parse()
    assert len(parsed) == 1
    assert "req1" in parsed
    routine = parsed["req1"]

    assert routine == RequestRoutine(
        name="req1",
        return_vars=[],
        callback_type=CallbackType.WAIT_ALL,
        callback=None,
        request=IqoalaRequest(
            name="req1",
            remote_id=Template("client_id"),
            epr_socket_id=0,
            num_pairs=3,
            virt_ids=RequestVirtIdMapping.from_str("custom 1, 2, 3"),
            timeout=1000,
            fidelity=0.65,
            typ=EprType.MEASURE_DIRECTLY,
            role=EprRole.RECEIVE,
            result_array_addr=0,
        ),
    )


def test_parse_multiple_request():
    text = """
REQUEST req1
  callback_type: wait_all
  callback: 
  return_vars: 
  remote_id: 1
  epr_socket_id: 0
  num_pairs: 5
  virt_ids: all 0
  timeout: 1000
  fidelity: 0.65
  typ: create_keep
  role: create
  result_array_addr: 3

REQUEST req2
  callback_type: sequential
  callback: subrt1
  return_vars: 
  remote_id: 1
  epr_socket_id: 0
  num_pairs: 3
  virt_ids: increment 1
  timeout: 1000
  fidelity: 0.65
  typ: measure_directly
  role: receive
  result_array_addr: 0
    """

    parsed = RequestRoutineParser(text).parse()
    assert len(parsed) == 2
    assert "req1" in parsed
    assert "req2" in parsed
    routine1 = parsed["req1"]
    routine2 = parsed["req2"]

    assert routine1 == RequestRoutine(
        name="req1",
        return_vars=[],
        callback_type=CallbackType.WAIT_ALL,
        callback=None,
        request=IqoalaRequest(
            name="req1",
            remote_id=1,
            epr_socket_id=0,
            num_pairs=5,
            virt_ids=RequestVirtIdMapping.from_str("all 0"),
            timeout=1000,
            fidelity=0.65,
            typ=EprType.CREATE_KEEP,
            role=EprRole.CREATE,
            result_array_addr=3,
        ),
    )
    assert routine2 == RequestRoutine(
        name="req2",
        return_vars=[],
        callback_type=CallbackType.SEQUENTIAL,
        callback="subrt1",
        request=IqoalaRequest(
            name="req2",
            remote_id=1,
            epr_socket_id=0,
            num_pairs=3,
            virt_ids=RequestVirtIdMapping.from_str("increment 1"),
            timeout=1000,
            fidelity=0.65,
            typ=EprType.MEASURE_DIRECTLY,
            role=EprRole.RECEIVE,
            result_array_addr=0,
        ),
    )


def test_parse_invalid_request():
    text = """
REQUEST req1
  callback_type: wait_all
  callback: 
  return_vars: 
  remote_id: 1
  epr_socket_id: 0
  num_pairs: 3
  virt_ids: all 0
  timeout: 1000
  fidelity: 0.65
  typ: invalid
  role: receive
  result_array_addr: 0
    """

    # invalid 'typ' value

    with pytest.raises(IqoalaParseError):
        RequestRoutineParser(text).parse()


DEFAULT_META = """
META_START
name: alice
parameters: 
csockets: 0 -> bob
epr_sockets: 
META_END
"""


def test_parse_program():
    meta_text = """
META_START
name: alice
parameters: 
csockets: 0 -> bob
epr_sockets: 
META_END
    """

    program_text = """
^b0 {type = CL}:
    my_value = assign_cval() : 1
    remote_id = assign_cval() : 0
    send_cmsg(remote_id, my_value)
    received_value = recv_cmsg(remote_id)
    new_value = assign_cval() : 3
    my_value = add_cval_c(new_value, new_value)

^b1 {type = QL}:
    vec<m> = run_subroutine(vec<my_value>) : subrt1

^b2 {type = QC}:
    vec<m> = run_request(vec<my_value>) : req1

^b3 {type = CL}:
    return_result(m)
    """

    subrt_text = """
SUBROUTINE subrt1
    params: my_value
    returns: m
    uses: 0
    keeps:
    request: 
  NETQASM_START
    set Q0 0
    rot_z Q0 {my_value} 4
    meas Q0 M0
    ret_reg M0
  NETQASM_END
    """

    req_text = """
REQUEST req1
  callback_type: wait_all
  callback: 
  return_vars: 
  remote_id: 1
  epr_socket_id: 0
  num_pairs: 3
  virt_ids: increment 1
  timeout: 1000
  fidelity: 0.65
  typ: measure_directly
  role: receive
  result_array_addr: 0
    """

    parsed_program = IqoalaParser(
        meta_text=meta_text,
        host_text=program_text,
        subrt_text=subrt_text,
        req_text=req_text,
    ).parse()
    assert len(parsed_program.blocks) == 4
    assert len(parsed_program.instructions) == 9
    assert "subrt1" in parsed_program.local_routines
    assert len(parsed_program.request_routines) == 1
    assert "req1" in parsed_program.request_routines


def test_parse_program_no_requests():
    meta_text = """
META_START
name: alice
parameters: 
csockets: 0 -> bob
epr_sockets: 
META_END
    """

    program_text = """
^b0 {type = CL}:
    my_value = assign_cval() : 1
    remote_id = assign_cval() : 0
    send_cmsg(remote_id, my_value)
    received_value = recv_cmsg(remote_id)
    new_value = assign_cval() : 3
    my_value = add_cval_c(new_value, new_value)

^b1 {type = QL}:
    vec<m> = run_subroutine(vec<my_value>) : subrt1

^b1 {type = CL}:
    return_result(m)
    """

    subrt_text = """
SUBROUTINE subrt1
    params: my_value
    returns: m
    uses: 0
    keeps:
    request: 
  NETQASM_START
    set Q0 0
    rot_z Q0 {my_value} 4
    meas Q0 M0
    ret_reg M0
  NETQASM_END
    """

    req_text = """
    """

    parsed_program = IqoalaParser(
        meta_text=meta_text,
        host_text=program_text,
        subrt_text=subrt_text,
        req_text=req_text,
    ).parse()
    assert len(parsed_program.blocks) == 3
    assert len(parsed_program.instructions) == 8
    assert "subrt1" in parsed_program.local_routines
    assert len(parsed_program.request_routines) == 0


def test_parse_program_no_subroutines():
    meta_text = """
META_START
name: alice
parameters: 
csockets: 0 -> bob
epr_sockets: 
META_END
    """

    program_text = """
^b0 {type = CL}:
    my_value = assign_cval() : 1
    remote_id = assign_cval() : 0
    send_cmsg(remote_id, my_value)
    received_value = recv_cmsg(remote_id)
    new_value = assign_cval() : 3
    m = add_cval_c(new_value, new_value)
    return_result(m)
    """

    subrt_text = """
    """

    req_text = """
    """

    parsed_program = IqoalaParser(
        meta_text=meta_text,
        host_text=program_text,
        subrt_text=subrt_text,
        req_text=req_text,
    ).parse()
    assert len(parsed_program.blocks) == 1
    assert len(parsed_program.instructions) == 7
    assert len(parsed_program.request_routines) == 0


def test_parse_program_invalid_subrt_reference():
    meta_text = """
META_START
name: alice
parameters: 
csockets: 0 -> bob
epr_sockets: 
META_END
    """

    program_text = """
^b0 {type = CL}:
    my_value = assign_cval() : 1
^b1 {type = QL}:
    vec<m> = run_subroutine(vec<my_value>) : non_existing_subrt
^b2 {type = CL}:
    return_result(m)
    """

    subrt_text = """
SUBROUTINE subrt1
    params: my_value
    returns: m
    uses: 
    keeps:
    request: 
  NETQASM_START
    set R0 0
  NETQASM_END
    """

    with pytest.raises(IqoalaParseError):
        IqoalaParser(
            meta_text=meta_text,
            host_text=program_text,
            subrt_text=subrt_text,
            req_text="",
        ).parse()


def test_parse_program_invalid_req_routine_reference():
    meta_text = """
META_START
name: alice
parameters: 
csockets: 0 -> bob
epr_sockets: 
META_END
    """

    program_text = """
^b0 {type = CL}:
    my_value = assign_cval() : 1
^b1 {type = QL}:
    vec<m> = run_request(vec<my_value>) : non_existing_req
^b2 {type = CL}:
    return_result(m)
    """

    subrt_text = """
SUBROUTINE subrt1
    params: my_value
    returns: m
    uses: 
    keeps:
    request: 
  NETQASM_START
    set R0 0
  NETQASM_END
    """

    with pytest.raises(IqoalaParseError):
        IqoalaParser(
            meta_text=meta_text,
            host_text=program_text,
            subrt_text=subrt_text,
            req_text="",
        ).parse()


def test_split_text():
    meta_text = """
META_START
name: alice
parameters: 
csockets: 0 -> bob
epr_sockets: 
META_END
    """

    host_text = """
^b0 {type = CL}:
    m = assign_cval() : 1
    return_result(m)
    """

    subrt_text = """
SUBROUTINE subrt1
    params: 
    returns: 
    request: 
  NETQASM_START
    set Q0 0
  NETQASM_END
    """

    req_text = """
REQUEST req1
  callback_type: wait_all
  callback:
  return_vars: 
  remote_id: {client_id}
  epr_socket_id: 0
  num_pairs: 1
  virt_ids: all 0
  timeout:1000
  fidelity: 1.0
  typ: create_keep
  role: receive
  result_array_addr: 0
    """

    text = meta_text + host_text + subrt_text + req_text
    parser = IqoalaParser(text)

    assert text_equal(parser._meta_text, meta_text)
    assert text_equal(parser._host_text, host_text)
    assert text_equal(parser._subrt_text, subrt_text)
    assert text_equal(parser._req_text, req_text)


def test_split_text_multiple_subroutines():
    meta_text = """
META_START
name: alice
parameters: 
csockets: 0 -> bob
epr_sockets: 
META_END
    """

    host_text = """
^b0 {type = CL}:
    m = assign_cval() : 1
    return_result(m)
    """

    subrt_text = """
SUBROUTINE subrt1
    params: 
    returns: 
    uses:
    keeps:
    request: 
  NETQASM_START
    set Q0 0
  NETQASM_END

SUBROUTINE subrt2
    params: 
    returns: 
    uses:
    keeps:
    request: 
  NETQASM_START
    set Q7 7
  NETQASM_END
    """

    text = meta_text + host_text + subrt_text
    parser = IqoalaParser(text)

    assert text_equal(parser._meta_text, meta_text)
    assert text_equal(parser._host_text, host_text)
    assert text_equal(parser._subrt_text, subrt_text)


def test_split_text_no_subroutines():
    meta_text = """
META_START
    name: server
    parameters: client_id
    csockets: 0 -> client
    epr_sockets: 0 -> client
META_END
    """

    host_text = """
    ^b0 {type = CL}:
        remote_id = assign_cval() : {client_id}
    ^b1 {type = QC}:
        run_request(vec<>) : req1
        """

    req_text = """
REQUEST req1
    callback_type: wait_all
    callback:
  return_vars: 
    remote_id: {client_id}
    epr_socket_id: 0
    num_pairs: 1
    virt_ids: all 0
    timeout:1000
    fidelity: 1.0
    typ: create_keep
    role: receive
    result_array_addr: 0
        """

    text = meta_text + host_text + req_text
    parser = IqoalaParser(text)

    assert text_equal(parser._meta_text, meta_text)
    assert text_equal(parser._host_text, host_text)
    assert text_equal(parser._req_text, req_text)


def test_parse_program_single_text():
    text = """
META_START
name: alice
parameters: 
csockets: 0 -> bob
epr_sockets: 
META_END

^b0 {type = CL}:
    my_value = assign_cval() : 1
    remote_id = assign_cval() : 0
    send_cmsg(remote_id, my_value)
    received_value = recv_cmsg(remote_id)
    new_value = assign_cval() : 3
    my_value = add_cval_c(new_value, new_value)
^b1 {type = QL}:
    vec<m> = run_subroutine(vec<my_value>) : subrt1
^b2 {type = CL}:
    return_result(m)

SUBROUTINE subrt1
    params: my_value
    returns: m
    uses: 0
    keeps:
    request: 
  NETQASM_START
    set Q0 0
    rot_z Q0 {my_value} 4
    meas Q0 M0
    ret_reg M0
  NETQASM_END
    """

    parsed_program = IqoalaParser(text).parse()
    assert len(parsed_program.blocks) == 3
    assert len(parsed_program.instructions) == 8
    assert "subrt1" in parsed_program.local_routines


def test_parse_file():
    path = os.path.join(os.path.dirname(__file__), "test_parse_server.iqoala")
    with open(path) as file:
        text = file.read()
    parsed_program = IqoalaParser(text).parse()
    assert len(parsed_program.blocks) == 9
    assert len(parsed_program.instructions) == 11
    assert "create_epr_0" in parsed_program.local_routines
    assert "create_epr_1" in parsed_program.local_routines
    assert "local_cphase" in parsed_program.local_routines
    assert "meas_qubit_0" in parsed_program.local_routines
    assert "meas_qubit_1" in parsed_program.local_routines
    assert "req0" in parsed_program.request_routines
    assert "req1" in parsed_program.request_routines
    assert parsed_program.local_routines["create_epr_0"].request_name == "req0"
    assert parsed_program.local_routines["create_epr_1"].request_name == "req1"


def test_parse_file_2():
    path = os.path.join(os.path.dirname(__file__), "test_parse_client.iqoala")
    with open(path) as file:
        text = file.read()
    parsed_program = IqoalaParser(text).parse()
    assert len(parsed_program.blocks) == 6
    assert len(parsed_program.instructions) == 19
    assert "create_epr_0" in parsed_program.local_routines
    assert "post_epr_0" in parsed_program.local_routines
    assert "create_epr_1" in parsed_program.local_routines
    assert "post_epr_1" in parsed_program.local_routines
    assert "req0" in parsed_program.request_routines
    assert "req1" in parsed_program.request_routines
    assert parsed_program.local_routines["create_epr_0"].request_name == "req0"
    assert parsed_program.local_routines["create_epr_1"].request_name == "req1"


if __name__ == "__main__":
    test_parse_incomplete_meta()
    test_parse_meta_no_end()
    test_parse_meta()
    test_parse_meta_multiple_remotes()
    test_parse_1_instr()
    test_parse_2_instr()
    test_parse_faulty_instr()
    test_parse_vec()
    test_parse_vec_2_elements()
    test_parse_vec_2_elements_and_return()
    test_parse_vec_2_elements_and_return_2_elements()
    test_parse_block_header()
    test_parse_block()
    test_get_block_texts()
    test_parse_multiple_blocks()
    test_parse_subrt()
    test_parse_subrt_2()
    test_parse_multiple_subrt()
    test_parse_invalid_subrt()
    test_parse_request()
    test_parse_request_2()
    test_parse_request_with_template()
    test_parse_multiple_request()
    test_parse_invalid_request()
    test_parse_program()
    test_parse_program_no_requests()
    test_parse_program_no_subroutines()
    test_parse_program_invalid_subrt_reference()
    test_parse_program_invalid_req_routine_reference()
    test_split_text()
    test_split_text_multiple_subroutines()
    test_split_text_no_subroutines()
    test_parse_program_single_text()
    test_parse_file()
    test_parse_file_2()
