import pickle

def save_resluts(pop, pset1, toolbox, hof,filename1='', filename_forHof=''):
    with open(filename1+'pop.pkl', 'wb') as f:
        pickle.dump(pop, f)
    with open(filename1+'pset.pkl' , 'wb') as f:
        pickle.dump(pset1, f)
    with open(filename1+'toolbox.pkl' , 'wb') as f:
        pickle.dump(toolbox, f)
def load_resluts(filename):
    with open(filename, 'rb') as f:
        loaded_pop = pickle.load(f)
        return loaded_pop