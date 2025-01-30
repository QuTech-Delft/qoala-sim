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
    alpha = 0.2 # attenuation in fiber
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
    # We divide by 20 instead of 10 since the heralded entanglement station is placed evenly between the two nodes
    exp_loss_term = 10**(-alpha * distance / 20)
    p_ent = eta_ion*eta_ion_FC*eta_telecom_det
    rate = 0.5 * eta_penalty * (p_ent**2 * exp_loss_term) / t_cycle
    
    # The time it takes to generate a single pair in seconds
    time_s = 1 / rate
    # Convert to nanoseconds
    time_ns = int(1e9 * time_s) 

    return time_ns, fidelity

def nv_epr(distance: float, optimism: int=0) -> tuple[int, float]:
    c = 200000 # Speed of light in fiber at telecom freq, km / s
    alpha = 0.05
    eta_fid = 1 #0.92 calculated
    p_emd = 5.1e-4
    t_cycle = 3.8e-6 # 3.8 microseconds
    t_com = distance / c

    if optimism == 1:
        p_emd = 0.48

    p_det = p_emd*10**(-0.2*distance / 20)
    p_ent = 2*p_det*alpha
    rate = p_ent / (t_cycle+t_com)

    fidelity = (1 - alpha)*eta_fid
    time_s = 1 / rate
    time_ns = int(1e9*time_s)
    return time_ns, fidelity

def cc_time(distance: float) -> int:
    """
    Returns the amount of time needed to send a classical message over a given distance
    :param distance: distance between nodes in km
    :return: the time it takes to send the message in ns
    """ 
    distance_nodeinfo_map = {
        0: {
            "Node0": "Delft 1",
            "Node1": "Delft 1",
            "distance": 0,
            "hops": 0,
        },
        2.2: {
            "Node0": "Delft 1",
            "Node1": "Delft 2",
            "distance": 2.2,
            "hops": 0,
        },
        16.8: {
            "Node0": "Delft 1",
            "Node1": "R'dam 1",
            "distance": 16.8,
            "hops": 1,
        },
        19.8: {
            "Node0": "Delft 1",
            "Node1": "Den Haag 2",
            "distance": 19.8,
            "hops": 0,
        },
        26.3: {
            "Node0": "Delft 1",
            "Node1": "Den Haag 1",
            "distance": 26.3,
            "hops": 1,
        },
        30.6: {
            "Node0": "Delft 1",
            "Node1": "Leiden 1",
            "distance": 30.6,
            "hops": 0,
        },
        33.1: {
            "Node0": "Delft 1",
            "Node1": "R'dam 2",
            "distance": 33.1,
            "hops": 1,
        },
        40.2: {
            "Node0": "Delft 1",
            "Node1": "R'dam 3",
            "distance": 40.2,
            "hops": 1,
        },
        47.9: {
            "Node0": "Delft 1",
            "Node1": "Leiden 2",
            "distance": 47.9,
            "hops": 2,
        },
        55.2: {
            "Node0": "Delft 1",
            "Node1": "Leiden 3",
            "distance": 55.2,
            "hops": 3,
        },
    }

    nodeinfo = distance_nodeinfo_map[distance] 
    h = nodeinfo["hops"]
    c = 200000 # Speed of light in fiber at telecom frequency, km/s
    D_ew = distance/c
    D_pt = h*244e-6 + 155e-6
    D_s = D_pt + D_ew
    D_ns = int(1e9*D_s)
    return D_ns