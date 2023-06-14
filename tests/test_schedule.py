from qoala.lang.ehi import EhiNetworkSchedule


def test_network_schedule():
    # [0, 100), [500, 600), [1000, 1100) etc
    schedule1 = EhiNetworkSchedule(bin_length=100, first_bin=0, bin_period=500)
    assert schedule1.next_bin(0) == 0
    assert schedule1.next_bin(10) == 500
    assert schedule1.next_bin(100) == 500
    assert schedule1.next_bin(200) == 500
    assert schedule1.next_bin(500) == 500
    assert schedule1.next_bin(510) == 1000

    # [150, 250), [350, 450), [550, 650) etc
    schedule1 = EhiNetworkSchedule(bin_length=100, first_bin=150, bin_period=200)
    assert schedule1.next_bin(0) == 150
    assert schedule1.next_bin(100) == 150
    assert schedule1.next_bin(150) == 150
    assert schedule1.next_bin(200) == 350
    assert schedule1.next_bin(250) == 350
    assert schedule1.next_bin(300) == 350
    assert schedule1.next_bin(350) == 350
    assert schedule1.next_bin(351) == 550


if __name__ == "__main__":
    test_network_schedule()
