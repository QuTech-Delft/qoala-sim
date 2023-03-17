class SharedMemoryRegion:
    pass


class SharedMemoryPool:
    def __init__(self) -> None:
        self._rr_in = SharedMemoryRegion()
        self._rr_out = SharedMemoryRegion()
        self._cr_in = SharedMemoryRegion()
        self._lr_in = SharedMemoryRegion()
        self._lr_out = SharedMemoryRegion()
