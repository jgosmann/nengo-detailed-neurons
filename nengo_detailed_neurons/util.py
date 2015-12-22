def nrn_duration(t):
    """Converts a duration in native Nengo units to native Neuron units."""
    # Nengo uses seconds
    # Neuron uses milliseconds
    return 1000 * t
