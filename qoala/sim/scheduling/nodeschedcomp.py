from __future__ import annotations

from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.component import Component, Port

from qoala.runtime.message import Message
from qoala.sim.scheduling.procsched import ProcessorScheduler


class NodeSchedulerComponent(Component):
    """
    NetSquid component representing for a node scheduler.
    It is used to send messages from the node scheduler to processor schedulers.

    :param name: Name of the component
    :param cpu_scheduler: CPU scheduler that node scheduler will send messages to.
    :param qpu_scheduler: QPU scheduler that node scheduler will send messages to.

    """

    def __init__(
        self,
        name,
        cpu_scheduler: ProcessorScheduler,
        qpu_scheduler: ProcessorScheduler,
        internal_sched_latency: float = 0.0,
    ):
        super().__init__(name=name)
        self.add_ports(["cpu_scheduler_out", "cpu_scheduler_in"])
        self.add_ports(["qpu_scheduler_out", "qpu_scheduler_in"])

        # Channel: NodeScheduler -> CPU scheduler
        node_sched_to_cpu = ClassicalChannel(
            "node_scheduler_to_cpu_scheduler", delay=internal_sched_latency
        )
        self.cpu_scheduler_out_port.connect(node_sched_to_cpu.ports["send"])
        node_sched_to_cpu.ports["recv"].connect(cpu_scheduler.node_scheduler_in_port)

        # Channel: NodeScheduler -> QPU scheduler
        node_sched_to_qpu = ClassicalChannel(
            "node_scheduler_to_qpu_scheduler", delay=internal_sched_latency
        )
        self.qpu_scheduler_out_port.connect(node_sched_to_qpu.ports["send"])
        node_sched_to_qpu.ports["recv"].connect(qpu_scheduler.node_scheduler_in_port)

        # Channel: CPU scheduler -> NodeScheduler
        cpu_to_node_sched = ClassicalChannel(
            "cpu_scheduler_to_node_scheduler", delay=internal_sched_latency
        )
        self.cpu_scheduler_in_port.connect(cpu_to_node_sched.ports["recv"])
        cpu_to_node_sched.ports["send"].connect(cpu_scheduler.node_scheduler_out_port)

        # Channel: QPU scheduler -> NodeScheduler
        qpu_to_node_sched = ClassicalChannel(
            "qpu_scheduler_to_node_scheduler", delay=internal_sched_latency
        )
        self.qpu_scheduler_in_port.connect(qpu_to_node_sched.ports["recv"])
        qpu_to_node_sched.ports["send"].connect(qpu_scheduler.node_scheduler_out_port)

    @property
    def cpu_scheduler_out_port(self) -> Port:
        """
        Port used to send messages to the CPU scheduler.
        """
        return self.ports["cpu_scheduler_out"]

    @property
    def qpu_scheduler_out_port(self) -> Port:
        """
        Port used to send messages to the QPU scheduler.
        """
        return self.ports["qpu_scheduler_out"]

    @property
    def cpu_scheduler_in_port(self) -> Port:
        """
        Port used to receive messages from the CPU scheduler.
        """
        return self.ports["cpu_scheduler_in"]

    @property
    def qpu_scheduler_in_port(self) -> Port:
        """
        Port used to receive messages from the QPU scheduler.
        """
        return self.ports["qpu_scheduler_in"]

    def send_cpu_scheduler_message(self, msg: Message) -> None:
        """
        Send a message to the CPU scheduler.
        :param msg: Message to send.
        :return: None
        """
        self.cpu_scheduler_out_port.tx_output(msg)

    def send_qpu_scheduler_message(self, msg: Message) -> None:
        """
        Send a message to the QPU scheduler.
        :param msg: Message to send.
        :return: None
        """
        self.qpu_scheduler_out_port.tx_output(msg)
