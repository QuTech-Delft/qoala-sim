"""Warning categories for :mod:`qoala.runtime.noise`.

This module must not import NetSquid (directly or through other Qoala modules).
pytest-xdist re-imports the module that defines a warning's category in order
to forward the warning from a worker to the controller, and it does so from a
non-main thread. Importing NetSquid there fails, because ``pydynaa`` installs a
signal handler on import, which is only allowed in the main thread.
"""


class PerQubitDepolarNoiseWarning(UserWarning):
    """Warns that NetSquid's per-qubit ``DepolarNoiseModel`` is used on a
    multi-qubit gate.

    On a multi-qubit gate that model applies one independent depolarizing
    channel per qubit, not the joint channel ``rho -> (1 - p) rho + p I/2**n``,
    so the gate's fidelity depends on its output state (see the docstring of
    :mod:`qoala.runtime.noise`). Configurations that select it explicitly
    (``GateDepolariseConfig``) keep working unchanged for backwards
    compatibility, e.g. to compare with earlier results. Use ``GateJointDepolariseConfig`` /
    :class:`qoala.runtime.noise.JointDepolarNoiseModel` for the joint channel.

    To silence it deliberately::

        import warnings
        from qoala.runtime.noise import PerQubitDepolarNoiseWarning
        warnings.filterwarnings("ignore", category=PerQubitDepolarNoiseWarning)
    """
