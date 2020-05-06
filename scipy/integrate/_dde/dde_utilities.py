import numpy as np


def discontinuityDetection(t0, tf, delays):
    """Discontinuity detection between t0 and tf.

    Parameters
        t0 ():
        tf ():
    ----------
    Returns
    -------
    nxtDisc : (int)
        index of the nearst discontinuity
    discont : ndarray, shape (nbr_discontinuities,)
        array with all discont within my interval of integration

    References
    ----------
    .. [1] S. Shampine, Thompson, "?????" dde23 MATLAB
    """

    discont = None

    #  discontinuites detection
    if not delays:
        discont = tf
        delayMin = np.inf
    else:

        inter_delays = delays # list of intersection of delays
        tmp_delays = delays

        while(tmp_delays):  # adding intersection between delays
            delay_k = tmp_delays[0]
            tmp_delays = tmp_delays[1:]
            inter_delays = inter_delays + (delay_k + np.asarray(tmp_delays)).tolist()
        del tmp_delays
        # definition of al delays intersections from delays summation
        discont = np.arange(t0, tf, delays[0])
        for tau_i in inter_delays[1:]:
            discont = np.append(discont, np.arange(t0, tf, tau_i))
        discont = np.asarray(sorted(set(discont)) + [tf])
        #  conbinaison of all delays + intersection of delays between t0 tf
        diff = np.append(np.array([10000.0]), np.diff(discont))
        #  addition of 1 element in array to do diff operation and keep same length
        discont = discont[~(diff < 1e-12)]  #  on enleves doublons
        nxtDisc = 1  # indice diiie la prochain discontinuite
    return nxtDisc, discont

