from qoala.lang.hostlang import (
    AssignCValueOp,
    BasicBlock,
    BasicBlockType,
    ReturnResultOp,
)
from qoala.util.tests import text_equal


def test_basic_block():
    block = BasicBlock(
        name="b0",
        typ=BasicBlockType.CL,
        instructions=[AssignCValueOp("value", 3), ReturnResultOp("value")],
    )

    expected = """
^b0 {type = CL}:
    value = assign_cval() : 3
    return_result(value)
    """

    assert text_equal(str(block), expected)


if __name__ == "__main__":
    test_basic_block()
