"""Quantum noise models used by Qoala on top of the ones provided by NetSquid.

Why this module exists
----------------------
NetSquid's :class:`~netsquid.components.models.qerrormodels.DepolarNoiseModel`
depolarizes every qubit an instruction acts on *independently*: on a two-qubit
gate it applies two single-qubit depolarizing channels, each with probability
``p``. It does **not** apply the joint two-qubit channel

    rho -> (1 - p) * rho + p * I/4.

Qoala, however, converts a configured two-qubit gate fidelity ``F`` into a
probability with :func:`qoala.util.math.fidelity_to_prob_max_mixed`, which
returns ``p = (1 - F) * 4/3``. That is the inverse of ``F = 1 - 3p/4``, the
fidelity left by the *joint* channel on any pure state. Combining that formula
with the per-qubit model makes the simulated gate noisier than configured, and
by an amount that depends on the state the gate produces:

=============  ==================================  ==============================
Output state   Per-qubit model (NetSquid)          Joint model (this module)
=============  ==================================  ==============================
Bell pair      ``F = 1 - 3p/2 + 3p^2/4``           ``F = 1 - 3p/4``
``|00>``       ``F = (1 - p/2)^2``                 ``F = 1 - 3p/4``
=============  ==================================  ==============================

For a configured ``F = 0.99`` a CNOT that prepares a Bell pair ended up with
``F = 0.9801`` (about twice the configured infidelity) and a gate with a
product-state output with ``F = 0.9867`` (about 4/3 of it).

:class:`JointDepolarNoiseModel` fixes this by applying the depolarizing channel
to all qubits of the instruction *jointly*, so every output state gets exactly
the configured fidelity and the existing ``p = (1 - F) * 4/3`` conversion is
correct.
"""

from __future__ import annotations

import functools
import itertools
import math
import warnings
from typing import Any, Dict, List, Sequence, Tuple, Type

from netsquid.components.models.qerrormodels import DepolarNoiseModel, QuantumErrorModel
from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits.qubit import Qubit

# Re-exported so it can be imported from here. It is *defined* in a module
# without NetSquid imports: see qoala.runtime.noise_warnings.
from qoala.runtime.noise_warnings import PerQubitDepolarNoiseWarning

__all__ = [
    "JointDepolarNoiseModel",
    "PerQubitDepolarNoiseWarning",
    "joint_depolarize",
    "warn_if_per_qubit_depolar_on_multi_qubit_gate",
]


@functools.lru_cache(maxsize=None)
def _pauli_basis(num_qubits: int) -> Tuple[ops.Operator, ...]:
    """Return all ``4**num_qubits`` tensor products of {I, X, Y, Z}.

    The first element is always the identity ``I ⊗ ... ⊗ I``.
    The result is cached since the same sizes are requested on every gate.
    """
    result: List[ops.Operator] = []
    for paulis in itertools.product((ops.I, ops.X, ops.Y, ops.Z), repeat=num_qubits):
        op = paulis[0]
        for p in paulis[1:]:
            op = op ^ p  # `^` is the tensor product in NetSquid.
        result.append(op)
    return tuple(result)


def joint_depolarize(qubits: Sequence[Qubit], prob: float) -> None:
    """Apply the joint ``n``-qubit depolarizing channel to ``qubits``.

    The channel is

        rho -> (1 - prob) * rho + prob * I / 2**n

    where ``n = len(qubits)``. It is implemented through its Pauli-twirl Kraus
    representation, using ``I / 2**n = 4**-n * sum_P P rho P`` over all
    ``n``-qubit Paulis ``P``:

    * the identity gets weight ``1 - prob + prob / 4**n``,
    * each of the other ``4**n - 1`` Paulis gets weight ``prob / 4**n``.

    For ``n == 1`` this is exactly :func:`netsquid.qubits.qubitapi.depolarize`.
    In a pure-state formalism (KET) NetSquid samples one of the Paulis with
    these weights instead of building the mixed state.

    :param qubits: The qubits to depolarize together. Their order does not
        matter, as the channel is permutation invariant.
    :param prob: Probability of replacing the joint state by the maximally
        mixed state. Must lie in ``[0, 1]``.
    """
    if prob < 0 or prob > 1:
        raise ValueError(f"{prob} is an invalid probability.")
    if math.isclose(prob, 0.0) or len(qubits) == 0:
        return

    num_qubits = len(qubits)
    paulis = _pauli_basis(num_qubits)
    p_each = prob / len(paulis)
    weights = [p_each] * len(paulis)
    weights[0] = 1.0 - prob + p_each
    qapi.multi_operate(list(qubits), list(paulis), weights)


class JointDepolarNoiseModel(DepolarNoiseModel):
    """Depolarizing noise applied **jointly** to all qubits of an instruction.

    Drop-in replacement for NetSquid's
    :class:`~netsquid.components.models.qerrormodels.DepolarNoiseModel`
    (same constructor arguments and properties). Only
    :meth:`error_operation` is overridden.

    For the ``n`` (non-``None``) qubits given to :meth:`error_operation`, the
    model applies

        rho -> (1 - p) * rho + p * I / 2**n

    instead of ``n`` independent single-qubit depolarizing channels. With a
    single qubit both models are identical. With two qubits every pure output
    state ``psi`` keeps fidelity ``<psi|rho|psi> = 1 - 3p/4``, which is what
    :func:`qoala.util.math.fidelity_to_prob_max_mixed` assumes. See the module
    docstring for the full explanation of why NetSquid's model does not match.

    The probability ``p`` is:

    * ``depolar_rate`` when ``time_independent`` is ``True`` (the mode Qoala
      uses for gate noise);
    * ``1 - exp(-delta_time * depolar_rate * 1e-9)`` otherwise, i.e. the same
      conversion NetSquid's
      :func:`~netsquid.qubits.qubitapi.delay_depolarize` uses, with
      ``delta_time`` in ns and ``depolar_rate`` in Hz.
    """

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Apply the joint depolarizing channel to ``qubits``.

        Overrides
        :meth:`netsquid.components.models.qerrormodels.DepolarNoiseModel.error_operation`,
        which applies one independent depolarizing channel per qubit.

        :param qubits: Qubits the instruction acted on. ``None`` entries
            (empty memory positions) are skipped, as in the parent class.
        :param delta_time: Time the qubits spent on the component [ns]. Only
            used if ``time_independent`` is ``False``.
        """
        present = [q for q in qubits if q is not None]
        if len(present) == 0:
            return

        if self.time_independent:
            prob = self.depolar_rate
        else:
            if self.depolar_rate < 0:
                raise ValueError(
                    f"depolar rate {self.depolar_rate} should be non-negative."
                )
            prob = 1.0 - math.exp(-delta_time * self.depolar_rate * 1e-9)

        joint_depolarize(present, prob)


def warn_if_per_qubit_depolar_on_multi_qubit_gate(
    instruction_name: str,
    num_qubits: int,
    error_model: Type[QuantumErrorModel],
    error_model_kwargs: Dict[str, Any],
) -> None:
    """Emit :class:`PerQubitDepolarNoiseWarning` if a multi-qubit gate uses
    NetSquid's per-qubit ``DepolarNoiseModel`` with a non-zero probability.

    No warning is emitted for:

    * single-qubit gates (per-qubit and joint channels are identical there);
    * noiseless gates (``depolar_rate == 0``, e.g. ``GateConfig.perfect_config``);
    * :class:`JointDepolarNoiseModel` or any other model (the check is on the
      exact class, not on subclasses).
    """
    if num_qubits < 2 or error_model is not DepolarNoiseModel:
        return
    if error_model_kwargs.get("depolar_rate", 0) == 0:
        return
    warnings.warn(
        f"Gate '{instruction_name}' acts on {num_qubits} qubits but uses "
        "NetSquid's DepolarNoiseModel (GateDepolariseConfig), which depolarizes "
        "each qubit independently rather than jointly. The gate's fidelity then "
        "depends on its output state (about 2x the expected infidelity on a "
        "Bell pair). Use GateJointDepolariseConfig (JointDepolarNoiseModel) for "
        "a joint depolarizing channel. See qoala.runtime.noise.",
        PerQubitDepolarNoiseWarning,
        stacklevel=3,
    )
