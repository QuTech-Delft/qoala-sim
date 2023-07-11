from qoala.lang.ehi import EhiNetworkSchedule, EhiNetworkTimebin


def test_network_schedule():
    # 0:    (1, 0, 2, 0)
    # 100:  (1, 1, 2, 1)
    # 200:  (1, 2, 2, 2)
    # 1000: (1, 0, 2, 0)
    # 1100: (1, 1, 2, 1)
    # 1200: (1, 2, 2, 2)
    # 2000: (1, 0, 2, 0)
    # 2100: (1, 1, 2, 1)
    # 2200: (1, 2, 2, 2)
    # etc.

    node1 = 1
    node2 = 2

    def bin(pid1: int, pid2: int) -> EhiNetworkTimebin:
        return EhiNetworkTimebin(node1, pid1, node2, pid2)

    pattern = [
        bin(0, 0),
        bin(1, 1),
        bin(2, 2),
    ]
    schedule1 = EhiNetworkSchedule(
        bin_length=100, first_bin=0, bin_pattern=pattern, repeat_period=1000
    )
    assert schedule1.next_bin(0) == (0, bin(0, 0))
    assert schedule1.next_bin(80) == (100, bin(1, 1))
    assert schedule1.next_bin(100) == (100, bin(1, 1))
    assert schedule1.next_bin(180) == (200, bin(2, 2))
    assert schedule1.next_bin(200) == (200, bin(2, 2))
    assert schedule1.next_bin(280) == (1000, bin(0, 0))
    assert schedule1.next_bin(900) == (1000, bin(0, 0))
    assert schedule1.next_bin(1000) == (1000, bin(0, 0))
    assert schedule1.next_bin(1080) == (1100, bin(1, 1))


if __name__ == "__main__":
    test_network_schedule()
