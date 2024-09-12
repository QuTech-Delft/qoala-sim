from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class MultiQubit:
    """
    A class that stores a list of qubit ids, which is used primarily for multi-qubit gates.

    :param qubit_ids: List of qubit ids. The order of the qubit ids is important. For example for the CNOT gate,
    the first qubit id is the control qubit and the second qubit id is the target qubit.
    """

    qubit_ids: List[int]

    def __hash__(self) -> int:
        """
        Hash function for the MultiQubit class. This is needed to use MultiQubit as a key in a dictionary.

        :return: Hash value of the MultiQubit object.
        """
        return hash(tuple(self.qubit_ids))
