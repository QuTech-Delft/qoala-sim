from pydynaa import EventType

EVENT_WAIT = EventType("SCHEDULER_WAIT", "scheduler wait")
EPR_DELIVERY = EventType("EPR_DELIVERY", "EPR delivery")


# Signals inside a single node
SIGNAL_HOST_HOST_MSG = "EvHostHostMsg"
SIGNAL_HOST_QNOS_MSG = "EvHostQnosMsg"
SIGNAL_HOST_NSTK_MSG = "EvHostNstkMsg"
SIGNAL_QNOS_HOST_MSG = "EvQnosHostMsg"
SIGNAL_QNOS_NSTK_MSG = "EvQnosNstkMsg"
SIGNAL_NSTK_HOST_MSG = "EvNstkHostMsg"
SIGNAL_NSTK_QNOS_MSG = "EvNstkQnosMsg"
SIGNAL_NSTK_NSTK_MSG = "EvNstkNstkMsg"

# Signals between a node and the EntDist
SIGNAL_NSTK_ENTD_MSG = "EvNstkEntdMsg"
SIGNAL_ENTD_NSTK_MSG = "EvEntdNstkMsg"

# Global signals
SIGNAL_MEMORY_FREED = "EvMemoryFreed"
SIGNAL_TASK_COMPLETED = "TaskCompleted"
MSG_REQUEST_DELIVERED = "RequestDelivered"
