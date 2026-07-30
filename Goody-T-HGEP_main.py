import random
import warnings
#from mpi4py import MPI
from data.readData import loadData
from operator import add,mul,truediv
from config.gepConfig import *
import numpy as np
from utils.fitness import RMSE_N
from utils.model_io import save_resluts
from utils.basicOperations import *
import sys
from pathlib import Path

# 当前脚本所在目录
ROOT = Path(__file__).resolve().parent

# 第三方库目录
GEP_LIB = ROOT / "gep"

# 添加所有第三方库
if GEP_LIB.exists():
    sys.path.insert(0, str(GEP_LIB))

import geppy as gep
from deap import base, creator, tools
from mpi4py import MPI


def master_process():
    ########################
    ########load data#######
    ########################
    datas=loadData()
    trainOmegas=datas['trainOmegas']
    trainPhis=datas['trainPhis']
    trainLayerParas=datas['trainLayerParas']
    datasForTrain=[trainOmegas,trainPhis,trainLayerParas]

    ##############################
    ########set pset of GEP#######
    ##############################
    pset = gep.PrimitiveSet('Main', input_names=[  'G', 'beta_star','H','der','cf'])  ###'delta_RC', 'eta_star'
    pset.add_function(add, 2)
    pset.add_function(mul, 2)
    pset.add_function(truediv, 2)
    pset.add_function(root, 1)
    pset.add_rnc_terminal()
    pset.add_pow_terminal('G')
    pset.add_pow_terminal('beta_star')
    pset.add_pow_terminal('H')
    pset.add_pow_terminal('der')
    pset.add_pow_terminal('cf')

    #################################################
    #######initialize individual & Fitness###########
    #################################################
    creator.create("FitnessMin", base.Fitness, weights=(-1,))  # to minimize the objective (fitness)
    creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

    ##################################
    ########set toolbos of GEP########
    ##################################
    toolbox = gep.Toolbox()
    toolbox.register('rnc_gen', random.choice, np.arange(-1, 7, 0.1))
    toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=lengtHead, rnc_gen=toolbox.rnc_gen, rnc_array_length=lengtHead*2+1)
    toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=linker_goody)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register('compile', gep.compile_goody, pset=pset)
    toolbox.register('select', tools.selTournament, k=round((2.0 / 3.0) * n_pop) + 1, tournsize=2)

    #set mutation operation
    toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb='4p', pb=0.05)
    toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.025)
    toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.025)
    toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.025)
    toolbox.register('cx_1p', gep.crossover_one_point, pb=0.025)
    toolbox.register('cx_2p', gep.crossover_two_point, pb=0.025)
    toolbox.register('cx_gene', gep.crossover_gene, pb=0.025)
    toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb='4p', pb=0.025)
    toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.05)
    toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.05)
    toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='4p', pb=0.05)

    #Fitness Function. Please modify RT_ID to represent the corresponding segment index of the model.
    toolbox.register('evaluate', RMSE_N, datas=datasForTrain, toolbox=toolbox,RT_ID=1)

    ##################################
    ########set state recorder########
    ##################################
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    Max_evolution = [[]]
    Hof_save = [[] ]

    #########################################
    ########Core evolutionary process########
    #########################################

    #Initialize the population
    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1)
    print('Population initialization completed. Starting evolutionary iterations.', flush=True)

    #Evolutionary iteration
    pop, log = gep.gep_simple_goody_MPI(pop, toolbox, n_gens=n_genes, n_generations=n_gen,
                                            n_elites=15, stats=stats,
                                            hall_of_fame=hof, verbose=True,
                                            optimizer=True, opt_period=2, opt_prob=0.001, opt_bounds=(-1, 5), comm=comm)

    #save the results
    Max_evolution[0] = log.select("min")
    Hof_save[0] = hof
    best_ind = Hof_save[0][0]
    list_reluts = []
    for i in range(n_genes):
        list_reluts.append(str(best_ind[i]))
    print(list_reluts)
    with open(rf'{savaPath}\best.txt', 'w') as f:
        f.write('[')
        for i in range(len(list_reluts)):
            f.write("'")
            f.write(list_reluts[i])
            f.write("'")

    save_resluts(pop, pset, toolbox, hof, filename1=rf'{savaPath}/',
                     filename_forHof='hof2D.txt')
    print('Best individual and population information saved successfully.', flush=True)
    comm.Abort()

def worker_process():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    opt_bounds = (-1, 5)
    from scipy.optimize import fmin_slsqp
    def Local_sub_func(RNC_array):
        """
        SubFucntion to be used within local optimizer
        input : RNC array to be optimize
        output: fitness to be minimized
        """
        # idx = np.random.random((1,opt_N))

        for j in range(len(ind[0][0].rnc_array)):
            ind_local_opt[Loc_Idx].rnc_array[j] = RNC_array[j]

        # fitness = toolbox.evaluate_opt(ind_local_opt,opt_Idx)
        fitness = toolbox.evaluate(ind_local_opt)
        return fitness[0]

    def Local_optimizer(Local_sub_func):
        """
        Local otpimizer based on the minimize function of scipy
        The objective is to allow the algorithm to find new numerical constant
        using the minimize scipy function instead of the random search of GEP
        """
        try:

            # new appmpt
            in_bounds = list(opt_bounds for i in range(len(ind_local_opt[Loc_Idx].rnc_array)))
            res = fmin_slsqp(Local_sub_func, ind_local_opt[Loc_Idx].rnc_array,
                             bounds=in_bounds,
                             iprint=-1)
            # res = minimize(Local_sub_func,ind_local_opt[Loc_Idx].rnc_array, bounds=bds)
            Best_RNC = res
            # print('Yes')

            # bds = (opt_bounds,)*len(population[0][0].rnc_array)
            # res = least_squares(Local_sub_func,ind_local_opt[Loc_Idx].rnc_array, bounds=opt_bounds, xtol=1e-3, max_nfev=10)
            # res = minimize(Local_sub_func,ind_local_opt[Loc_Idx].rnc_array, bounds=bds)
            # Best_RNC = res.x
            # print('Yes')
        except:
            Best_RNC = ind_local_opt[0].rnc_array
            flag_err = +1
            #print('\r number of Error in local optimizer : %i ' % (flag_err), sep=' ', end='', flush=True)
            # print('number of Error in local optimizer')
        return Best_RNC
    while True:
        # 向主进程请求任务
        comm.send(None, dest=0, tag=11)
        #print(f"Process {rank} requested a task",flush=True)

        # 接收任务
        status = MPI.Status()
        task = comm.recv(source=0, tag=12, status=status)
        if task==None:
            continue
        toolbox = comm.recv(source=0, tag=13, status=status)  #一定要保证每一条send都能被成功rec，不然发送信息的主进程会阻塞
        if type(task)==tuple:
            opt_prob=0.5
            index,ind=task

            if np.random.uniform() < opt_prob:
                # print('ind opt')
                #flag_opt =  1
                global ind_local_opt

                ind_local_opt = ind
                global Loc_Idx
                Loc_Idx = 0
                for j in range(len(ind)):  # n_genes
                    Best_RNC = Local_optimizer(Local_sub_func)
                    # print(Best_RNC)
                    for k in range(len(ind[0].rnc_array)):  # len RNC array
                        # elites[i][j].rnc_array[k] = Best_RNC[k]
                        ind[j].rnc_array[k] = Best_RNC[k]
                    Loc_Idx = Loc_Idx + 1
                del ind.fitness.values

            comm.send(ind,dest=0,tag=15)


        else:
            result =toolbox.evaluate(task)
            comm.send((task,result), dest=0, tag=14)


if __name__ == '__main__':
    # 全局忽略RuntimeWarning（不推荐，可能掩盖其他问题）
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        master_process()
    else:
        worker_process()

