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


if __name__ == "__main__":
    test_virt_id_mapping_to_string()
    test_string_to_virt_id_mapping()
