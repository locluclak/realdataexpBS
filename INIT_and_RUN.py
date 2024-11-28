import pivot
import pivot_nonDA
import numpy as np
import time

import _heartfailure
import _seoulbike
import _walmart
import _co2
global seed
def run(iter = 0):    
    seed = int(np.random.rand() * (2**32 - 1))
    # seed = 457262911   
    # print("Seed:",seed)
    ns = 100
    nt = 20

    Xs, Ys = _heartfailure.larger50(ns)
    Xt, Yt = _heartfailure.undereq50(nt)

    # Xs, Ys = _seoulbike.no_holiday(ns)
    # Xt, Yt = _seoulbike.holiday(nt)

    # Xs, Ys = _walmart.Walmart_sales_noholiday(ns)
    # Xt, Yt = _walmart.Walmart_sales_holiday(nt)

    # Xs, Ys = _co2.other_fuel(ns)
    # Xt, Yt = _co2.gasoline_fuel(nt)
    p = Xs.shape[1]

    Bs = np.dot(np.dot(np.linalg.inv(np.dot(Xs.T, Xs)), Xs.T) , Ys)
    try:
        # avoid singular matrix
        Bt = np.dot(np.dot(np.linalg.inv(np.dot(Xt.T, Xt)), Xt.T) , Yt)
    except:
        Xt = Xt + 0.0005*np.identity(nt)
        Bt = np.dot(np.dot(np.linalg.inv(np.dot(Xt.T, Xt)), Xt.T) , Yt)
    Ys_ = Xs.dot(Bs)
    Yt_ = Xt.dot(Bt)
    var_s = 1/(ns - p) * (Ys - Ys_).T.dot(Ys - Ys_)
    var_t = 1/(nt - p) * (Yt - Yt_).T.dot(Yt - Yt_)

    Sigma_s = np.identity(ns) * var_s 
    Sigma_t = np.identity(nt) * var_t 
    #___________________________________________________________

    # betat = 4
    # true_beta_s = np.full((p,1), 2) #source's beta
    # true_beta_t = np.full((p,1), betat) #target's beta
    k = 3 # k=-1 if choose based criterion
    #___________________________________________________________

    pvalue = pivot.pvalue_SI(seed, ns, nt, p, k, Xs, Xt, Ys, Yt, Sigma_s, Sigma_t, 'DS')

    # pvalue = pivot_nonDA.pvalue_SI(seed, nt, p, Xt, Yt, Sigma_t)

    # Save pvalue into file
    # OCorPARA_FIXorAIC_FPRorTPR = 'para_AIC_time'
    # filename = f'Experiment/Listpvalue_{OCorPARA_FIXorAIC_FPRorTPR}_{ns}_{p}.txt'
    # filename = f'Experiment/Listpvalue_{OCorPARA_FIXorAIC_FPRorTPR}_{ns}_{p}_{betat}.txt'
    # with open(filename, 'a') as f:
    #     f.write(str(en-st)+ '\n')
    return pvalue

if __name__ == "__main__":
    for i in range(130):
        # st = time.time()
        print(f'{i}.')
        print(run())
        # en = time.time()
        # print(f"Time of 1 pvalue {i}: {en - st}")