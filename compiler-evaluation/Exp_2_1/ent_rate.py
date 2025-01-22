# A function that returns the rate, fidelity of entanglement generation
def trapped_ion_epr(distance: float, QIA_SGA:int=0) -> tuple[int, float]:
    """
    Calculates the time it takes to generate an EPR pair and the fidelity of the pair given the distance between nodes and the parameter regime being used.
    :param distance: The distance between the two nodes in km
    :param QIA_SGA: Takes a value of 0,1,2. 0 uses current realized parameters for trapped ions, 1 and 2 use optimistic parameter values hoped to be realized by QIA in SGA 1 and 2 respectively.
    :return: (link_duration, fidelity), the time it takes to generate an EPR pair in ns, and the fidelity of the generated pair
    """
    # Parameter values taken from data sheet given by Soubhadra
    c = 200000 # Speed of light in fiber at telecom frequency km/s
    alpha = 0.2
    eta_ion = 0.462 / 0.87
    eta_ion_FC = 0.25
    eta_telecom_det = 0.75
    t_class = distance / c
    t_prep = 1.8e-3 #1.8ms
    eta_penalty = 0.12
    fidelity = 0.88
    
    # QIA SGA 1 optimistic values
    if QIA_SGA == 1:
        eta_ion = 0.5 / 0.87
        eta_ion_FC = 0.5
        t_prep = 1e-3 #1ms
    # QIA SGA 2 optimistic values
    elif QIA_SGA == 2:
        eta_ion = 0.5 / 0.87
        eta_ion_FC = 0.7
        eta_telecom_det = 0.9
        t_prep = 200e-6 # 200 microseconds
        eta_penalty = 0.2
        fidelity = 0.95

    t_cycle = t_class + t_prep
    
    # Equations come from Parameter Definition document given by Soubhadra 
    exp_loss_term = 10**(-alpha * distance / 10)
    p_ent = eta_ion*eta_ion_FC*eta_telecom_det
    rate = 0.5 * eta_penalty * (p_ent**2 * exp_loss_term) / t_cycle

    # The time it takes to generate a single pair in seconds
    time_s = 1 / rate
    # Convert to nanoseconds
    time_ns = int(1e9 * time_s) 

    return time_ns, fidelity

def cc_time(distance: float) -> int:
    """
    Returns the amount of time needed to send a classical message over a given distance via TCP.
    :param distance: distance between nodes in km
    :return: the time it takes to send the message in ns
    """
    # c = 200000 # Speed of light in fiber at telecom frequency, km/s
    # time_s = 2.1 * distance / c
    # time_ns = int(time_s* 1e9)
    if distance < 1:
        return 1e5 #0.1ms
    else:
        return 8e6 #8ms
