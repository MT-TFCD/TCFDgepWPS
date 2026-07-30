# coding=utf-8
"""
.. moduleauthor:: Shuhua Gao

This :mod:`basic` module provides fundamental boilerplate GEP algorithm implementations. After registering proper
operations into a :class:`deap.base.Toolbox` object, the GEP evolution can be simply launched using the present
algorithms. Of course, for complicated problems, you may want to define your own algorithms, and the implementation here
can be used as a reference.
"""
import deap
import random
import warnings
import numpy as np
from scipy.optimize import minimize, least_squares, fmin_slsqp
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.dummy import Pool as ThreadPool
import time
import threading

def _validate_basic_toolbox(tb):
    """
    Validate the operators in the toolbox *tb* according to our conventions.
    """
    assert hasattr(tb, 'select'), "The toolbox must have a 'select' operator."
    # whether the ops in .pbs are all registered
    for op in tb.pbs:
        assert op.startswith('mut') or op.startswith('cx'), "Operators must start with 'mut' or 'cx' except selection."
        assert hasattr(tb, op), "Probability for a operator called '{}' is specified, but this operator is not " \
                                "registered in the toolbox.".format(op)
    # whether all the mut_ and cx_ operators have their probabilities assigned in .pbs
    for op in [attr for attr in dir(tb) if attr.startswith('mut') or attr.startswith('cx')]:
        if op not in tb.pbs:
            warnings.warn('{0} is registered, but its probability is NOT assigned in Toolbox.pbs. '
                          'By default, the probability is ZERO and the operator {0} will NOT be applied.'.format(op),
                          category=UserWarning)


def _apply_modification(population, operator, pb):
    """
    Apply the modification given by *operator* to each individual in *population* with probability *pb* in place.
    """
    for i in range(len(population)):
        if random.random() < pb:
            population[i], = operator(population[i])
            del population[i].fitness.values
    return population


def _apply_crossover(population, operator, pb):
    """
    Mate the *population* in place using *operator* with probability *pb*.
    """
    for i in range(1, len(population), 2):
        if random.random() < pb:
            population[i - 1], population[i] = operator(population[i - 1], population[i])
            del population[i - 1].fitness.values
            del population[i].fitness.values
    return population


def gep_simple(population, toolbox, n_generations=100, n_elites=1,
               stats=None, hall_of_fame=None, verbose=__debug__):
    """
    This algorithm performs the simplest and standard gene expression programming.
    The flowchart of this algorithm can be found
    `here <https://www.gepsoft.com/gxpt4kb/Chapter06/Section1/SS1.htm>`_.
    Refer to Chapter 3 of [FC2006]_ to learn more about this basic algorithm.

    .. note::
        The algorithm framework also supports the GEP-RNC algorithm, which evolves genes with an additional Dc domain for
        random numerical constant manipulation. To adopt :func:`gep_simple` for GEP-RNC evolution, use the
        :class:`~geppy.core.entity.GeneDc` objects as the genes and register Dc-specific operators.
        A detailed example of GEP-RNC can be found at `numerical expression inference with GEP-RNC
        <https://github.com/ShuhuaGao/geppy/blob/master/examples/sr/numerical_expression_inference-RNC.ipynb>`_.
        Users can refer to Chapter 5 of [FC2006]_ to get familiar with the GEP-RNC theory.

    :param population: a list of individuals
    :param toolbox: :class:`~geppy.tools.toolbox.Toolbox`, a container of operators. Regarding the conventions of
        operator design and registration, please refer to :ref:`convention`.
    :param n_generations: max number of generations to be evolved
    :param n_elites: number of elites to be cloned to next generation
    :param stats: a :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param hall_of_fame: a :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: whether or not to print the statistics.
    :returns: The final population
    :returns: A :class:`~deap.tools.Logbook` recording the statistics of the
              evolution process

    .. note:
        To implement the GEP-RNC algorithm for numerical constant evolution, the :class:`geppy.core.entity.GeneDc` genes
        should be used. Specific operators are used to evolve the Dc domain of :class:`~geppy.core.entity.GeneDc` genes
        including Dc-specific mutation/inversion/transposition and direct mutation of the RNC array associated with
        each gene. These operators should be registered into the *toolbox*.
    """
    _validate_basic_toolbox(toolbox)
    logbook = deap.tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(n_generations + 1):
        # evaluate: only evaluate the invalid ones, i.e., no need to reevaluate the unchanged ones
        invalid_individuals = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
        
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit

        # record statistics and log
        if hall_of_fame is not None:
            hall_of_fame.update(population)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_individuals), **record)
        
        # verbose
        if gen % 1 == 0:
            if verbose:
                #print(logbook.stream[gen])
                print(logbook[-1])
        
        # end condition
        if gen == n_generations:
            break

        # selection with elitism
        elites = deap.tools.selBest(population, k=n_elites)
        #offspring = toolbox.select(population, len(population) - n_elites)
        offspring = toolbox.select(population)
        
        # replication
        offspring = [toolbox.clone(ind) for ind in offspring]

        # mutation
        for op in toolbox.pbs:
            if op.startswith('mut'):
                offspring = _apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])

        # crossover
        for op in toolbox.pbs:
            if op.startswith('cx'):
                offspring = _apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])

        # replace the current population with the offsprings
        population = elites + offspring
        
    return population, logbook


def gep_simple_goody_MPI(population, toolbox, n_generations=100, n_elites=1,
                     stats=None, hall_of_fame=None, verbose=__debug__,
                     optimizer=None, opt_prob=0.25, opt_bounds=(0, 10), opt_period=1, comm=None, **kwargs):
    from mpi4py import MPI
    """
    Written by Joachim Dominique
    This algorithm performs the improved version of GEP with optimized search for numerical valuess and standard gene expression programming.
    """

    def Local_sub_func(RNC_array):
        """
        SubFucntion to be used within local optimizer
        input : RNC array to be optimize
        output: fitness to be minimized
        """
        # idx = np.random.random((1,opt_N))

        for j in range(len(population[0][0].rnc_array)):
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
            print('\r number of Error in local optimizer : %i ' % (flag_err), sep=' ', end='', flush=True)
            # print('number of Error in local optimizer')
        return Best_RNC

    _validate_basic_toolbox(toolbox)
    logbook = deap.tools.Logbook()
    logbook.header = ['gen', 'nevals', 'nan', 'nopts'] + (stats.fields if stats else [])

    flag_opt = 0
    for gen in range(n_generations + 1):
        flag_nan = 0
        # print("pop begin :%i" %(len(population)))
        # evaluate: only evaluate the invalid ones, i.e., no need to reevaluate the unchanged ones
        invalid_individuals = [ind for ind in population if not ind.fitness.valid]
        valid_individuals = [ind for ind in population if ind.fitness.valid]
        
        start_time = time.time()
        print(f'第{gen}代评估开始', flush=True)
        task_queue = invalid_individuals[:]
        num_workers = comm.Get_size() - 1
        active_workers = num_workers
        results = []
        received_count = 0
        total_to_receive = len(invalid_individuals)

        # 只要还有任务没分完，或者还有 Worker 没收到停止信号，就继续循环
        while received_count < total_to_receive or active_workers > 0:
            status = MPI.Status()
            # 允许接收任何来源、任何标签的消息
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()

            if tag == 14:
                # 情况 A：收到的是结果（评估模式返回 tag=14）
                results.append(msg)
                received_count += 1

            # elif tag == 15:
            #     # 情况 B：收到的是优化后的个体（优化模式返回 tag=15）
            #     results.append(msg)
            #     received_count += 1

            elif tag == 11:
                # 情况 C：收到的是 Worker 的请求
                if task_queue:
                    # 还有任务：派发任务 (tag=12) 和 工具箱 (tag=13)
                    task = task_queue.pop(0)
                    comm.send(task, dest=source, tag=12)
                    comm.send(toolbox, dest=source, tag=13)
                else:
                    # 任务发完了：发送结束指令，并减少活跃 Worker 计数
                    comm.send(None, dest=source, tag=12)
                    active_workers -= 1
                    # print(f"Worker {source} 领到 None，已下班")
        print('评估完成',flush=True)
        # for _ in range(len(invalid_individuals)):
        #     result = comm.recv(source=MPI.ANY_SOURCE, tag=14)
        #     results.append(result)


        #for ind, fit in zip(invalid_individuals, fitnesses):
            #ind.fitness.values = fit
            #except AssertionError:
                #print(fit,type(fit))
                #input()

            #if fit[0] > 1e50:
                #flag_nan = flag_nan + 1
        pop_MPI=[]
        for i in range(len(results)):
            ind=results[i][0]
            fit=results[i][1]
            ind.fitness.values=fit
            pop_MPI.append(ind)
        population=valid_individuals+pop_MPI
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"评估时间：{elapsed_time:.6f} 秒", flush=True)

        # record statistics and log
        if hall_of_fame is not None:
            hall_of_fame.update(population)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_individuals), nans=flag_nan, nopts=flag_opt, **record)

        # print results every 10 gen
        if gen % 1 == 0:
            if verbose:
                # print(logbook.stream[gen])
                print(logbook[-1])

        # end of algorithm condition
        if gen == n_generations:
            break

        # elistims selection
        elites = deap.tools.selBest(population, k=n_elites)
        # print(elites[0].fitness)


        # selection
        offspring_select = toolbox.select(population)
        offspring_select = [toolbox.clone(ind) for ind in offspring_select]

        # replication
        N_replicate = len(population) - len(offspring_select) - len(elites)
        randarray = np.random.randint(len(offspring_select), size=N_replicate)
        offspring_replicate = [toolbox.clone(offspring_select[rand]) for rand in randarray]
        # offspring_replicate = [toolbox.clone(population[rand]) for rand in randarray]

        # assembly
        offspring = offspring_select + offspring_replicate

        # optimizer
        # opt_N= 100
        cond1 = optimizer == True
        cond2 = gen % opt_period == 0
        cond3 = gen != 0

        flag_opt = 0
        global flag_err
        flag_err = 0
        if cond1 * cond2 * cond3:
            start_time = time.time()
            print('Start Local optimizer loop', flush=True)

            # global opt_Idx
            # opt_Idx = np.random.uniform(low=0, high=1, size=(opt_N,))
            # for i, ind in enumerate(invalid_individuals):
            #print(f"输入offspring长度{len(offspring)}")
            task_queue=[(i,offspring[i]) for i in range(len(offspring))]
            results = []
            while task_queue:
                status = MPI.Status()
                request = comm.recv(source=MPI.ANY_SOURCE, tag=11, status=status)
                source = status.Get_source()
                # print(f"Received request from process {source}",flush=True)
                # 分配一个任务
                # print(f'star send taks to {source}')
                if task_queue:
                    # print(f'send taks to {source}')
                    task = task_queue.pop(0)
                    comm.send(task, dest=source, tag=12)
                    comm.send(toolbox, dest=source, tag=13)
                else:
                    comm.send(None, dest=source, tag=12)
            #for _ in range(len(invalid_individuals)):
            while len(results)!=len(offspring):
                result = comm.recv(source=MPI.ANY_SOURCE, tag=15)
                results.append(result)
            offspring=results

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'局部优化时间：{elapsed_time:.6f} 秒', flush=True)
            #print(f'offspring长度：{len(offspring)}')
            print('End Local optimizer loop', flush=True)

        # mutation
        for op in toolbox.pbs:
            if op.startswith('mut'):
                offspring = _apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])

        # crossover
        for op in toolbox.pbs:
            if op.startswith('cx'):
                offspring = _apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])

        # replace the current population with the offsprings
        population = elites + offspring
        #print('种群长度：',len(population))

    return population, logbook




__all__ = ['gep_simple','gep_simple_goody_MPI']