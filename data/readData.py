import numpy as np
import pickle

def loadData(dataPath=r'E:\TCFDgepWPS\data\data_delta_star.pkl'):
    """
        Load training and testing datasets from a pickle file.

        Parameters
        ----------
        dataPath : str, optional
            Path to the pickle file. Default is 'data_delta_star.pkl'.

        Returns
        -------
        dataDict : dict
            Dictionary containing the following datasets:

            - testOmegas : list of ndarray
                Frequency parameters of the test dataset.
            - testPhis : list of ndarray
                Wavenumber parameters of the test dataset.
            - testLayerParas : list of ndarray
                Boundary-layer parameters of the test dataset.
                [RT,beta_star,G,H,Delta_ZS,cf]
            - testDataNames : list
                Names of the test datasets.

            - trainOmegas : list of ndarray
                Frequency parameters of the training dataset.
            - trainPhis : list of ndarray
                Wavenumber parameters of the training dataset.
            - trainLayerParas : list of ndarray
                Boundary-layer parameters of the training dataset.
                [RT,beta_star,G,H,Delta_ZS,cf]
            - trainDataNames : list
                Names of the training datasets.
        """
    
    with open(dataPath, 'rb') as f:
        dataKeys = ['testOmegas', 'testPhis', 'testLayerParas', 'testDataNames', 'trainOmegas', 'trainPhis',
                    'trainLayerParas', 'trainDataNames']
        dataDict = dict()
        rawPkl = pickle.load(f)
        rawKeys = list(rawPkl.keys())

        for index in range(len(dataKeys)):
            dataDict[dataKeys[index]] = rawPkl[rawKeys[index]]

    return dataDict

