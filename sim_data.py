import numpy as np
from solver import *

# c~n*m, M~n*k*m, d~k, A~n*l*m, b~n*l
def sim_data(n, m=10, k=10, l=10, c_bound=10, M_bound=10, A_bound=10, seed=42, use_seed=True):
    if use_seed:
    	np.random.seed(seed)

    c = np.random.uniform(c_bound, size=(n, m))
    M = np.random.uniform(M_bound, size=(n, k, m))
    # TODO: check bound
    d = np.random.uniform(n * m * M_bound / 4, n * m * M_bound * 3 / 4, size=(k))
    A = np.random.uniform(A_bound, size=(n, l, m))
    b = np.random.uniform(m * A_bound / 4, m * A_bound * 3 / 4, size=(n, l))
    return c, M, A, d, b

def sim_once(n, maxit=1000, alpha=1e-4, strategy="IAAL", tol=1e-3, use_seed=True):
	# TODO: default values except n
	c, M, A, d, b = sim_data(n, use_seed=use_seed)

	if strategy != "default":
		argmin_x, lambda_star, optimal_obj = primal_dual_solver(-c, M, A, d, b)

	if strategy == "IAAL":
		decay = True
		ascent_type = "full"
	elif strategy == "ADMM":
		decay = False
		ascent_type = "vr"
	primal_err_list = []
	lambd_list = [] 
	lambda_err_list = []
	duality_gap_list = []

	s_time = time.time()
	if strategy == "default":
		rs = general_basic_solver(-c, M, A, d, b)
	else:
		rs = general_st_sampling_solver(c, M, A, d, b, epoch=maxit, batch_size=1, alpha=alpha,
	                               decay=decay, decay_func=sqrt_decay,
	                               augmented=True, rho=1, prox_type=strategy,
	                               ascent_type=ascent_type, vr_m_order=2,
	                               err_list=primal_err_list, lambd_list=lambd_list, time_hist=[], 
	                               lambda_err_list=lambda_err_list, duality_gap_list=duality_gap_list,
	                               is_rep=False, is_cyc=False,
	                               answer=[], tol=tol, 
	                               optimal_obj=optimal_obj, lambda_star=lambda_star)
	e_time = time.time()
	total_time = e_time - s_time

	return primal_err_list, lambda_err_list, duality_gap_list, total_time

def sim_multi(n_range, maxit=200, alpha=1e-4, tol=1e-3, seed=42, num_avg=5):
	np.random.seed(seed)
	default_time_list, IAAL_time_list, ADMM_time_list = [], [], []
	for n in n_range:
		print("--------")
		print("size = {}".format(n))
		default_time, IAAL_time, ADMM_time = 0, 0, 0
		for i in range(num_avg):
			default_time += sim_once(n, maxit=maxit, alpha=alpha, strategy="default", 
				tol=tol, use_seed=False)[3]
			IAAL_time += sim_once(n, maxit=maxit, alpha=alpha, strategy="IAAL", 
				tol=tol, use_seed=False)[3]
			ADMM_time += sim_once(n, maxit=maxit, alpha=alpha, strategy="ADMM", 
				tol=tol, use_seed=False)[3]
		default_time_list.append(default_time / num_avg)
		IAAL_time_list.append(IAAL_time / num_avg)
		ADMM_time_list.append(ADMM_time / num_avg)
	return default_time_list, IAAL_time_list, ADMM_time_list
