from results.bestIndividual import gen_lists
from utils.jointCor import jointCor
from operator import add,mul,truediv
from utils.basicOperations import *
import numpy as np

def compile_goodyType(gen_list,omega,layer):
    """
        Compile a GEP individual into the Goody-type wall-pressure spectrum model.

        This function converts the symbolic expressions encoded in a GEP individual
        into executable Python expressions, and evaluates the corresponding model
        output based on the given frequency parameter and boundary-layer parameters.

        Parameters
        ----------
        gen_list : list
            A list of string expressions representing the genes of a GEP individual.
            Each element corresponds to one symbolic sub-function in the final model.

        omega :  narray
            Dimensionless angular frequency input.

        layer : tuple
            Boundary-layer parameters:
            (RT, beta_star, G, H, Delta_ZS, cf)

        Returns
        -------
        phi_GEP :array-like
            Predicted wall-pressure spectrum obtained from the GEP model.

        Notes
        -----
        The GEP genes are stored as string expressions and parsed using the
        ``eval()`` function. Therefore, the symbolic representation of individuals
        and the corresponding parsing function must strictly maintain identical
        variable names, operators, and symbols. Any modification to the symbolic
        format may result in parsing errors or incorrect model evaluation.
        """

    gen_compile = []
    RT,beta_star,G,H,Delta_ZS,cf = layer
    for i in range(len(gen_list)):
        gen_compile.append(eval(gen_list[i]))
    f1 = gen_compile[0];
    f2 = gen_compile[1];
    f3 = gen_compile[2];
    f4 = gen_compile[3];
    f5 = gen_compile[4];
    f6 = gen_compile[5];
    f7 = gen_compile[6];
    f8 = gen_compile[7];
    f9 = gen_compile[8]
    phi_GEP = (f1 * (omega) ** f2) / (((f9 * omega ** f3 + f4) ** f5) + (f6 * RT ** f7 * omega) ** f8)
    return phi_GEP

def solveHGEP(omega,layer,gen_lists):
    """
        Solve the hierarchical GEP (HGEP) model by combining multiple sub-models.

        The HGEP model consists of five Goody-type GEP sub-models stored in
        ``gen_lists``. Each sub-model is evaluated independently and combined
        using the corresponding joint correlation coefficients.

        Parameters
        ----------
        omega : float or array-like
            Dimensionless angular frequency input.

        layer : tuple
            Boundary-layer parameters required by the GEP sub-models.

        gen_lists : list
            A list containing five GEP sub-models of HGEP, where each sub-model
            is represented by its corresponding gene expressions.

        Returns
        -------
        phi_joint : float or array-like
            Combined wall-pressure spectrum predicted by the HGEP model.
        """

    RT=layer[0]
    phi_joint = 0
    jointCors = jointCor(RT)
    for i in range(len(gen_lists)):
        phi_i = compile_goodyType(gen_lists[i], omega, layer)
        phi_joint += phi_i * jointCors[i]


    return phi_joint

