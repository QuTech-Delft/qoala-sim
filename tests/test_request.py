from netqasm.lang.operand import Template

from qoala.lang.parse import RequestRoutineParser
from qoala.lang.request import RequestVirtIdMapping, VirtIdMappingType


def test_virt_id_mapping_to_string():
    all_zero = RequestVirtIdMapping(
        typ=VirtIdMappingType.EQUAL, single_value=0, custom_values=None
    )
    assert str(all_zero) == "all 0"

    all_three = RequestVirtIdMapping(
        typ=VirtIdMappingType.EQUAL, single_value=3, custom_values=None
    )
    assert str(all_three) == "all 3"

    increment_one = RequestVirtIdMapping(
        typ=VirtIdMappingType.INCREMENT, single_value=1, custom_values=None
    )
    assert str(increment_one) == "increment 1"

    custom = RequestVirtIdMapping(
        typ=VirtIdMappingType.CUSTOM, single_value=None, custom_values=[1, 2, 5]
    )
    assert str(custom) == "custom 1, 2, 5"

    all_template = RequestVirtIdMapping(
        typ=VirtIdMappingType.EQUAL,
        single_value=Template("virt_id"),
        custom_values=None,
    )
    assert str(all_template) == "all {virt_id}"

    increment_template = RequestVirtIdMapping(
        typ=VirtIdMappingType.INCREMENT,
        single_value=Template("virt_id"),
        custom_values=None,
    )
    assert str(increment_template) == "increment {virt_id}"


def test_string_to_virt_id_mapping():
    all_zero = RequestVirtIdMapping(
        typ=VirtIdMappingType.EQUAL, single_value=0, custom_values=None
    )
    assert RequestVirtIdMapping.from_str("all 0") == all_zero

    all_three = RequestVirtIdMapping(
        typ=VirtIdMappingType.EQUAL, single_value=3, custom_values=None
    )
    assert RequestVirtIdMapping.from_str("all 3") == all_three

    increment_one = RequestVirtIdMapping(
        typ=VirtIdMappingType.INCREMENT, single_value=1, custom_values=None
    )
    assert RequestVirtIdMapping.from_str("increment 1") == increment_one

    custom = RequestVirtIdMapping(
        typ=VirtIdMappingType.CUSTOM, single_value=None, custom_values=[1, 2, 5]
    )
    assert RequestVirtIdMapping.from_str("custom 1, 2, 5") == custom

    all_template = RequestVirtIdMapping(
        typ=VirtIdMappingType.EQUAL,
        single_value=Template("virt_id"),
        custom_values=None,
    )
    assert RequestVirtIdMapping.from_str("all {virt_id}") == all_template

    increment_template = RequestVirtIdMapping(
        typ=VirtIdMappingType.INCREMENT,
        single_value=Template("virt_id"),
        custom_values=None,
    )
    assert RequestVirtIdMapping.from_str("increment {virt_id}") == increment_template


def test_instantiate():
    text = """
REQUEST req1
  callback_type: wait_all
  callback: 
  return_vars: 
  remote_id: {client_id}
  epr_socket_id: 0
  num_pairs: {num_pairs}
  virt_ids: custom 1, 2, 3
  timeout: 1000
  fidelity: 0.65
  typ: measure_directly
  role: receive
    """

    request_routine = RequestRoutineParser(text).parse()["req1"]
    request = request_routine.request
    assert request.remote_id == Template(name="client_id")
    assert request.num_pairs == Template(name="num_pairs")

    request.instantiate(values={"client_id": 2, "num_pairs": 10})
    assert request.remote_id == 2
    assert request.num_pairs == 10


if __name__ == "__main__":
    test_virt_id_mapping_to_string()
    test_string_to_virt_id_mapping()
    test_instantiate()
