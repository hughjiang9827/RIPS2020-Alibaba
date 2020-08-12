from scipy.optimize import linprog
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import time
from qpsolvers import solve_qp
import cvxpy as cp

def get_indx_matrix(m, indices):
    return np.array([i * m + np.arange(m) for i in indices])

def make_PD(P, x=1e-7):
    diag = x * np.identity(len(P))
    return P + diag

# c~n*m, M~n*k*m, d~k, A~n*l*m, b~n*l
def primal_dual_solver(c, M, A, d, b):
    # TODO: check
    n, l, m = A.shape
    # TODO: do it somewhere else
    # c = -c
    c = c.flatten()

    A_ub_sep = block_diag(*A)
    # TODO: shape of M, d, b, lambd
    b = b.flatten()

    A_ub_couple = np.concatenate(M, axis=1)

    x = cp.Variable(n * m)
    constraints = [A_ub_couple @ x <= d, A_ub_sep @ x <= b, 0 <= x]
    objective = cp.Minimize(cp.sum(c.T @ x))
    prob = cp.Problem(objective, constraints)
    prob.solve()

    print("status: {}, optimal value: {}".format(prob.status, prob.value))
    argmin_x = x.value
    lambda_star = constraints[0].dual_value
    optimal_obj = prob.value

    return argmin_x, lambda_star, optimal_obj
    
# c~n*m, M~n*k*m, d~k, A~n*l*m, b~n*l
def general_basic_solver(c, M, A, d, b, is_coupled=True, augmented=False, P=None):
    # TODO: check
    n, l, m = A.shape
    # TODO: do it somewhere else
    # c = -c
    c = c.flatten()

    A_ub_sep = block_diag(*A)

    # TODO: shape of M, d, b, lambd
    b = b.flatten()
    A_ub, b_ub = A_ub_sep, b

    if is_coupled:
        k = M.shape[1]
        A_ub_couple = np.concatenate(M, axis=1)
        A_ub = np.concatenate((A_ub_couple, A_ub_sep), axis=0)
        b_ub = np.zeros(len(d) + len(b))
        b_ub[: len(d)] = d
        b_ub[len(d):] = b

    if augmented:
        # TODO: check
#         P = np.concatenate(P, axis=0)
#         P = block_diag(*P)
#         print(P)
        # TODO: make P PD
        argmin_x = solve_qp(make_PD(P), c, A_ub, b_ub, lb=np.zeros(len(c)))
    else:
        res = linprog(c, A_ub, b_ub)
        # TODO:
        # print(res.status)
#         print(np.linalg.matrix_rank(A_ub), A_ub.shape)
        s_check = res.status
        if s_check != 0:
            print("warning: {}".format(s_check))
        argmin_x = res.x

    return argmin_x

# c~n*m, M~n*k*m, d~k, A~n*l*m, b~n*l, lambd~k
def general_sub_problem_solver(c, M, A, d, b, sub_indices, lambd, augmented, dot_cache, w, rho=1):
    # TODO: min cj^Txj + lambda^T Mjxj + rho / 2 * ||dot_cache' - d + Mj xj||^2
    # rho / 2 * ||dot_cache - d + Mj xj||^2 = rho / 2 * ||w + Mj xj||^2 =
    # rho / 2 * (wT + xjkT MjkT)(w + Mjk xjk) ~ rho / 2 * (2wT Mjk xjk + xjkT MjkT Mjk xjk)
    # TODO: concatenate Mj Mk into Mj:k
    # TODO: save M[sub_indices] and M_2d
    c_sub = c[sub_indices] + np.matmul(lambd.T, M[sub_indices])
    A_sub = A[sub_indices]
    b_sub = b[sub_indices]

    P_sub = None
    if augmented:
#         w = dot_cache - d
        # TODO: check
        c_sub += rho * np.matmul(w.T, M[sub_indices])
        M_sub_2d = np.concatenate(M[sub_indices], axis=1)
#         P_sub = rho * np.matmul(np.transpose(M[sub_indices], axes=(0,2,1)), M[sub_indices])
        # TODO: check
        P_sub = rho / 2 * np.matmul(M_sub_2d.T, M_sub_2d)

    argmin_x_sub = general_basic_solver(c_sub, M, A_sub, d, b_sub, is_coupled=False,
                                        augmented=augmented, P=P_sub)

    return argmin_x_sub

# c~n*m, M~n*k*m, d~k, A~n*l*m, b~n*l, lambd~k
def general_st_sampling_solver(c, M, A, d, b, epoch=10, batch_size=1, alpha=1e-3,
                               decay=False, decay_func=None,
                               augmented=False, rho=1, prox_type="IAAL",
                               ascent_type="full", vr_m_order=2,
                               err_list=[], lambd_list=[], time_hist=[], 
                               lambda_err_list=[], duality_gap_list=[],
                               is_rep=True, is_cyc=False,
                               answer=[], tol=1e-3, optimal_obj=-1, lambda_star=[], 
                               max_check=[]):
    # TODO: check
    n, k, m = M.shape
    # convert max to min problem
    c = -c

    # TODO: check
#     c, M, A, d, b = [np.float_(dt) for dt in [c, M, A, d, b]]

    # # TODO: keep track of error
    # if len(answer) == 0:
    #     answer = general_basic_solver(c, M, A, d, b)

    # initialize lambda and argmin_x
    lambd, argmin_x = np.zeros(k), np.zeros(n * m)
    # TODO
    M_2d = np.concatenate(M, axis=1)
    c_1d = c.flatten()
    # cache
    dot_cache = np.zeros(k)
    current_obj = 0

    # TODO: VR
    dot_cache_t_prev = np.zeros(k)
    argmin_x_t_prev = np.zeros(n * m)

    # set seed
    seed = 42
    np.random.seed(seed)

    # counter
    counter = 0

    # TODO: keep track of runtime
    start_time = time.time()
    for s in range(epoch):
        if s == epoch - 1:
            print("Warning: maxit reached " + prox_type)
            max_check.append("max")

        # TODO: sample with or without rep?
        if is_rep:
            idx = np.random.randint(n, size=n)
        else:
            idx = np.random.permutation(n)

        # TODO: check
        if is_cyc:
            idx = np.arange(n)

        # TODO: VR
        if s % vr_m_order == 0:
            dot_cache_t = dot_cache_t_prev
            argmin_x_t = argmin_x_t_prev

        for j in range(0, n, batch_size):
            # counter
            if counter % 10000 == 0:
                print("{}/{}".format(counter, epoch * n / batch_size))

            sub_indices = idx[j: j + batch_size]
            update_indices = get_indx_matrix(m, sub_indices).flatten()

            # TODO: check
            # remove old values from cache
#             print(counter)
#             print(argmin_x)
#             print(dot_cache - d)
#             print("---")
            prev_Mx = np.matmul(M_2d[:, update_indices], argmin_x[update_indices])
            dot_cache -= prev_Mx
            current_obj -= np.dot(c_1d[update_indices], argmin_x[update_indices])

            # TODO: decay
            decayed_alpha = alpha
#             if s > 230:
#                 decay = True
            if decay:
                decayed_alpha = decay_func(alpha, s)
#                 print(decayed_alpha)

            w = 0
            if prox_type=="IAAL":
                w = dot_cache - d
            elif prox_type=="ADMM":
                w = (dot_cache - d) / n - (n - 1) * prev_Mx / n

            # TODO: check
#             rho = decayed_alpha
#             if s > 200:
#                 rho *= 10
            argmin_x_sub = general_sub_problem_solver(c, M, A, d, b, sub_indices, lambd,
                                                      augmented,
                                                      dot_cache,
                                                      # TODO: check,
                                                      w,
                                                      rho=decayed_alpha)

            # update cache
            argmin_x[update_indices] = argmin_x_sub
            # TODO: check
            d_psi = np.matmul(M_2d[:, update_indices], argmin_x[update_indices])
            dot_cache += d_psi
            current_obj += np.dot(c_1d[update_indices], argmin_x[update_indices])

            # TODO: check
            # lambd += alpha * (dot_cache - d)
            # TODO: VR
            lambd_prev = lambd
            if ascent_type == "vr":
#                 - batch_size * d / n
                d_psi_t = np.matmul(M_2d[:, update_indices], argmin_x_t[update_indices])
#                 - batch_size * d / n
                # TODO: check
                mu_t = batch_size * (dot_cache_t - d) / n
#               lambd = np.maximum(lambd + decayed_alpha * (d_psi), 0)
                lambd = np.maximum(lambd + decayed_alpha * (d_psi - d_psi_t + mu_t), 0)
#               lambd = np.maximum(lambd + decayed_alpha * (dot_cache - d - d_psi_t + mu_t), 0)
            elif ascent_type == "full":
                lambd = np.maximum(lambd + decayed_alpha * (dot_cache - d), 0)
            elif ascent_type == "single":
                lambd = np.maximum(lambd + decayed_alpha * (d_psi - batch_size * d / n), 0)

            # TODO: stopping criterion
#             if np.linalg.norm(lambd - lambd_prev) < tol:
#                 print(dot_cache - d)
#                 break

            # counter
            counter += 1
            end_time = time.time()

            # # TODO: keep track of error every iter
            # err = abs(optimal_obj - current_obj) / abs(optimal_obj)
            # # TODO: Termination
            # if all(dot_cache <= d) and err < tol:
            #     # break
            #     print(current_obj)
            #     return argmin_x

        # TODO: inner or outer loop
        # TODO: keep track of runtime
        time_hist.append(end_time - start_time)

        # TODO: keep track of error and lambd
        # err = np.linalg.norm(argmin_x - answer)
        err = abs(optimal_obj - current_obj) / abs(optimal_obj)
        err_list.append(err)
        # TODO: check
        lambd_list.append(lambd)
#          print(lambd)
        lambda_err = np.linalg.norm(lambd - lambda_star) / np.linalg.norm(lambda_star)
        lambda_err_list.append(lambda_err) 
        duality_gap = abs(np.dot(lambd.T, dot_cache - d))
        duality_gap_list.append(duality_gap)
        # TODO: Termination
        if all(dot_cache <= d) and err < tol:
            break

        # TODO: VR
        if s % vr_m_order == 0:
            dot_cache_t_prev = dot_cache
            argmin_x_t_prev = argmin_x

    print(current_obj)
    return argmin_x

def plot_error(err_list, name="Error", required=[]):
    err_list = np.array(err_list)

    plt.figure()
    plt.xlabel('Number of Epochs')
    plt.ylabel(name)
    plt.title(name)

    if name == "lambda" and err_list.shape[1] > 1:
        err_list_T = err_list.T
        if required == []:
            required = range(err_list.shape[1])
        for i in required:
            # TODO: check
            plt.plot(err_list_T[i], label=name + str(i))
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        plt.plot(err_list)

def log_decay(alpha, k):
    return alpha / (np.log(k + 1) + 1)

def sqrt_decay(alpha, k):
    return alpha / (np.sqrt(k) + 1)
