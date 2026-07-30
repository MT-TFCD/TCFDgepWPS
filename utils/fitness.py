import numpy as np
from utils.jointCor import jointCor

def RMSE_N(individual, datas,toolbox,RT_ID=1):
    """
        Calculate the normalized RMSE-based fitness for a GEP individual.

        The fitness is evaluated over multiple datasets. For each dataset.
        The final error is weighted by a factor according to the RT segment.

        Parameters
        ----------
        individual : object
            A GEP individual representing the symbolic expression to be evaluated.
        datas : tuple
            Input dataset containing:

            - omegas : list of numpy.ndarray
                Frequency parameters of each dataset.
            - phis : list of numpy.ndarray
                Target wall-pressure spectra of each dataset.
            - layers : list of numpy.ndarray
                Boundary-layer parameters of each dataset.

        toolbox : object
            GEP toolbox used to compile the GEP individual into an executable
            prediction function.

        RT_ID : int, optional
            Index of the RT segment used for Gaussian weighting. The RT range is
            divided into several segments, and RT_ID indicates the segment to which
            the current dataset belongs. Default is 1.

        Returns
        -------
        tuple
            Fitness value of the individual in the form required by DEAP.
            A smaller value indicates better prediction performance.
        """

    omegas,phis,layers=datas
    err = []

    for i in range( len(omegas)):
        omega = omegas[i]
        phi = phis[i]
        layer = layers[i]
        RT,beta_star,G,H,Delta_ZS,cf= layer
        try:
            f = toolbox.compile(individual, omega, RT)
            phi_pre = f(   G, beta_star,H,Delta_ZS,cf)
            mse = np.sqrt(np.mean((phi_pre - phi) ** 2))
            err_temp = mse/np.max(phi)

            if np.isnan(err_temp) or np.isinf(err_temp) or type(err_temp) == np.complex128:
                return 1e99,

            gaussFactor = jointCor(RT)[RT_ID-1]
            err_temp = err_temp * gaussFactor
            err.append(err_temp)


        except Exception as e:
            return 1e99,



    fitness =np.mean(np.array(err))


    return (fitness,)