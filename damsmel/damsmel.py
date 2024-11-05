# BISMILLAHIRRAHMANIRRAHIM
# First creation date   : 10/24/2024
# Last update           : 11/06/2024

"""
DAMSMEL: Directional Adaptive Metric Sampling Minimal Expected Loss

Alhamdulillah, DAMSMEL Python implementation has been completed in
this first stage. DAMSMEL is a continuous optimisation method
implementing iterative minimal expected loss of sampled points with
exponentially decaying distance. This algorithm has been proven to
converge to a global minimum in convex setting. For non-convex
setting, DAMSMEL tends to converge to a global minimum according to
our empirical tests compared to gradient-based optimisers.

This algorithm is developed in collaboration with Dr. Dieky Adzkiya
(Department of Mathematics, Institut Teknologi Sepuluh Nopember).
While the code implementation and testing are conducted by Rizal
Purnawan.

License: MIT
"""

# Necessary Libraries (Dependencies)
import time
import pandas as pd
import numpy as np

# DirectioNSMELRegressor
# -------------------------------------------------------------------
class DAMSMELRegressor:
    """
    Initialization Details
    ----------------------
    delta0          :   A real number representing the initial
                        distance of adjacent of sampling points.
    s               :   An integer, representing the number of
                        samples at each basis direction of
                        exploration.
    alpha           :   A real number, typically a number in between
                        0 and 1.
    M               :   Maximum number of iteration.
    fit_intercept   :   A boolean value, representing wether the
                        intercept in the regression model is to be
                        fitted or not.
    auto_truncate   :   A boolean value, executing iteration
                        truncation if True.
    """
    def __init__(
            self,
            delta0,
            s= 10,
            alpha= 0.1,
            M= 200,
            fit_intercept= False,
            auto_truncate= False,
            min_iter= 100
            ):
        self.delta0 = delta0
        self.alpha = alpha
        self.s = s
        self.M = M
        self.fit_intercept = fit_intercept
        self.auto_truncate = auto_truncate
        self.min_iter = min_iter


        self.weights = None
        self.monitor = None

    # OBJECTIVE FUNCTIONS
    # ---------------------------------------------------------------
    def mean_absolute_error(self, X, Y):
        """
        Description
        -----------
        'X' and 'Y' shall be a list (or numpy array) of the same
        length containing numeric values.
        """
        return np.mean(np.abs(np.array(X) - np.array(Y)))
    
    def squared_error(self, X, Y):
        """
        Description
        -----------
        'X' and 'Y' shall be a list (or numpy array) of the same
        length containing numeric values.
        """
        return np.sum((np.array(X) - np.array(Y))**2)

    def mean_squared_error(self, X, Y):
        """
        Description
        -----------
        'X' and 'Y' shall be a list (or numpy array) of the same
        length containing numeric values.
        """
        return np.mean((np.array(X) - np.array(Y))**2)

    def root_mean_squared_error(self, X, Y):
        """
        Description
        -----------
        'X' and 'Y' shall be a list (or numpy array) of the same
        length containing numeric values.
        """
        return np.sqrt(self.mean_squared_error(X, Y))

    def mean_absolute_percentage_error(X, Y):
        """
        Description
        -----------
        'X' and 'Y' shall be a list (or numpy array) of the same
        length containing numeric values.
        """
        return np.mean(
            [abs((x - y)/x) for x, y in zip(list(X), list(Y))]
            )

    def symmetric_mean_absolute_percentage_error(X, Y):
        """
        Description
        -----------
        'X' and 'Y' shall be a list (or numpy array) of the same
        length containing numeric values.
        """
        return np.mean(
            [
                abs(x - y)/(abs(x) + abs(y))
                for x, y in zip(list(X), list(Y))
                ]
            )

    def rnorm(self, X, Y):
        """
        Description
        -----------
        'X' and 'Y' shall be a list (or numpy array) of the same
        length containing numeric values.
        """
        X, Y = np.array(X), np.array(Y)
        EX = np.array([np.mean(X)] *len(X))
        norm = np.linalg.norm
        return (norm(X - Y) / norm(X - EX) )**2

    def map_obj(self, objtive_func= "MAE"):
        map = {
            "MAE": self.mean_absolute_error,
            "SE": self.squared_error,
            "MSE": self.mean_squared_error,
            "RMSE": self.root_mean_squared_error,
            "MAPE": self.mean_absolute_percentage_error,
            "SMAPE": self.symmetric_mean_absolute_percentage_error,
            "RNORM": self.rnorm
        }
        return map[objtive_func]
    
    
    # MAIN ALGORITHMS
    # ---------------------------------------------------------------
    def fit(
            self, X, y,
            u0= None,
            basis= None,
            d= 5,
            r= 0,
            objective_func= "MAE",
            runtime= True,
            verbose= False
            ):
        """
        Description
        -----------
        u0              :   Initial point of iteration, must be
                            either a numpy array or a list.
        basis           :   A list of numpy arrays representinf the
                            basis of Banach space of the search
                            space.
        d               :   An integer representing the either the
                            criterion of convergence or distance
                            reset.
        r               :   A small nonnegative real number
                            representing the criterion for
                            convergence or distance reset.
        objectve_func   :   A string representing the choice of
                            objective function. Current compatible
                            objective functions:
                            "MAE", "SE", "MSE", "RMSE", "MAPE",
                            "SMAPE", "RNORM"
        runtime         :   A boolean value, print the total runtime
                            if True.
        verbose         :   A boolean value, prints iteration
                            runtimes if True.
        """
        if self.fit_intercept == True:
            dim = len(X.columns) + 1
        else:
            dim = len(X.columns)
        if u0 is None:
            u0 = np.array([0] *dim)
        u_i = np.array(u0)
        if basis is None:
            basis = [np.array([0] *dim) for k in range(dim)]
            for k in range(dim):
                basis[k][k] = 1

        ell = self.map_obj(objective_func)
        norm = np.linalg.norm
        delta0, alpha, s, M = self.delta0, self.alpha, self.s, self.M

        R = list()
        R_ell = list()

        if self.fit_intercept == True:
            X_new = pd.DataFrame({"const": [1] *len(X)})
            for col in X.columns:
                X_new[col] = X[col]
            X_array = np.array(X_new)
        else:
            X_array = np.array(X.copy())

        rho = 0
        i = 1

        start = time.time()
        while i <= M:
            loop_start = time.time()

            if i > d:
                if all(norm(w - R[-1]) <= r for w in R[-d:]):
                    if self.auto_truncate == True \
                            and i > self.min_iter:
                        break
                    rho = i
            delta_i = delta0 *np.e**(-alpha *(i - rho))

            U_ij_list = [
                [u_i + (k *delta_i *b) for k in range(1, s + 1)]
                for b in basis
            ] + [
                [u_i + (-1 *k *delta_i *b) for k in range(1, s + 1)]
                for b in basis
            ]

            ind_dict = {
                k: np.mean(
                    [ell(
                        np.array(y),
                        np.array([np.dot(x, u) for u in X_array])
                        )
                    for x in U_ij_list[k]
                    ]
                )
                for k in range(len(U_ij_list))
            }
            j = min(ind_dict, key= ind_dict.get)

            candidates = {
                tuple(x): ell(
                    y, np.array(
                        [np.dot(np.array(x), u) for u in X_array]
                    )
                )
                for x in U_ij_list[j]
            }
            u_i = np.array(min(candidates, key= candidates.get))

            R.append(u_i)
            R_ell.append(np.min(list(candidates.values())))

            loop_end = time.time()
            if verbose == True:
                print(
                    f".....Loop {i}: {loop_end - loop_start} seconds"
                )

            i += 1

        finish = time.time()
        if runtime == True:
            if verbose == True:
                print("\n")
            print(f">>> Runtime: {finish - start} seconds []")
        monitor_df = pd.DataFrame({"solution": R, "loss": R_ell})

        weights = list(
            monitor_df[
                monitor_df["loss"] == monitor_df["loss"].min()
                ]["solution"]
            )[0]

        self.monitor = monitor_df.copy()
        self.weights = weights

    def predict(self, X):
        if self.weights is not None:
            weights = self.weights
            if self.fit_intercept == True:
                X_new = pd.DataFrame({"const": [1] *len(X)})
                for col in X.columns:
                    X_new[col] = X[col]
            else:
                X_new = X.copy()
            X_array = np.array(X_new)
            return np.array(
                [np.dot(weights, u) for u in X_array]
                )
        else:
            print("ERROR: Please train the model first!")
            raise ValueError
    

# -------------------------------------------------------------------
# DAMSMEL OPTIMIZER:
class DAMSMEL:
    """
    Initialization Details
    ----------------------
    basis           :   A list of numpy arrays representing the basis
                        for the Banach space in use.
    fit_intercept   :   A boolean value, representing wether the
                        intercept in the regression model is to be
                        fitted or not.
    auto_truncate   :   A boolean value, executing iteration
                        truncation if True.
    min_iter        :   An integer representig the minimum number of
                        iterations.
    """
    def __init__(
            self, Psi, dim,
            basis= None,
            auto_truncate= False,
            min_iter= 100
            ):
        self.Psi = Psi
        if not any(isinstance(dim, t) for t in [int, float]) \
                and basis is None:
            print("ERROR: Invalid 'dim' and 'basis' values!")
            raise ValueError
        if any(isinstance(dim, t) for t in [int, float]) \
                and basis is None:
            dim = int(dim) if isinstance(dim, float) else dim
            basis = [
                    np.array([0] *dim)
                    for k in range(dim)
                ]
            for k in range(dim):
                basis[k][k] = 1
        self.basis = basis
        self.auto_truncate = auto_truncate
        self.min_iter = min_iter

        self.monitor = None
        self.solution = None
    
    def optimize(
            self, u, delta0,
            alpha= 0.1, d= 5, s= 10,
            r= 0,
            M= 200,
            runtime= True,
            verbose= False
            ):
        """
        u       :   A numpy array or list representing the starting
                    point of iteration.
        delta0  :   A real number representing the initial distance
                    of adjacent of sampling points.
        alpha   :   A real number, typically a number in between
                    0 and 1.
        d       :   An integer representing the either the criterion
                    of convergence or distance reset.
        r       :   A small nonnegative real number representing the
                    criterion for convergence or distance reset.
        M       :   An integer representing the number of iterations.
        runtime :   A boolean value, print the total runtime if True.
        verbose :   A boolean value, prints iteration runtimes if True.
        """
        Psi = self.Psi
        basis = self.basis
        R = list()
        Psi_R = list()
        rho = 0
        u_i = np.array(u)
        i = 1
        norm = np.linalg.norm

        start = time.time()
        while i <= M:
            loop_start = time.time()
            if i > d:
                if all(norm(w - R[-1]) <= r for w in R[-d:]):
                    if self.auto_truncate == True and i > self.min_iter:
                        break
                    rho = i
            delta_i = delta0 *np.e**(-alpha *(i - rho))
            U_ij_list = [
                [u_i + (k *delta_i *b) for k in range(1, s + 1)]
                for b in basis
            ] + [
                [u_i + (-1 *k *delta_i *b) for k in range(1, s + 1)]
                for b in basis
            ]
            ind_dict = {
                k: np.mean([Psi(x) for x in U_ij_list[k]])
                for k in range(len(U_ij_list))
            }
            j = min(ind_dict, key= ind_dict.get)
            u_i = [
                x for x in U_ij_list[j]
                if Psi(x) == np.min([Psi(w) for w in U_ij_list[j]])
                ][0]
            R.append(u_i)
            Psi_R.append(Psi(u_i))

            loop_end = time.time()
            if verbose == True:
                print(f".....Loop {i}: {loop_end - loop_start} seconds")
            i += 1

        finish = time.time()
        if runtime == True:
            if verbose == True:
                print("\n")
            print(f">>> Runtime: {finish - start} seconds []")
        
        monitor_df = pd.DataFrame({"solution": R, "loss": Psi_R})

        v = list(
            monitor_df[
                monitor_df["loss"] == monitor_df["loss"].min()
                ]["solution"]
            )[0]
        
        self.monitor = monitor_df.copy()
        self.solution = v
        
        return v