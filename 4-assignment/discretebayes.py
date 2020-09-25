from typing import Tuple

import numpy as np


def discrete_bayes(
    # the prior: shape=(n,)
    pr: np.ndarray,
    # the conditional/likelihood: shape=(n, m)
    cond_pr: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray
]:  # the new marginal and conditional: shapes=((m,), (m, n))
    """Swap which discrete variable is the marginal and conditional."""
    
    # joint: p(x,z) = p(z|x)*p(z)
    joint = cond_pr * pr.reshape((-1, 1)) 
        
    # marginal: (probability that we are in more s_k given the previous measurements) p(z)
    marginal = cond_pr.T @ pr 
    
    # Take care of rare cases of degenerate zero marginal,
    conditional = joint / marginal 
    
    # flip axes?? (n, m) -> (m, n)
    conditional = conditional.T
    
    print_discrete_bayes = False
    if print_discrete_bayes:
        print("discrete_bayes():")
        print("cond_pr", cond_pr)
        print("pr", pr)
        print("pr.reshape((-1,1))", pr.reshape((-1,1)))
        print("joint:", joint, np.shape(joint))
        print("marginal:", marginal, np.shape(marginal))
        print("conditional:", conditional, np.shape(conditional))
    
    # optional DEBUG
    assert np.all(
        np.isfinite(conditional)
    ), f"NaN or inf in conditional in discrete bayes"
    assert np.all(
        np.less_equal(0, conditional)
    ), f"Negative values for conditional in discrete bayes"
    assert np.all(
        np.less_equal(conditional, 1)
    ), f"Value more than on in discrete bayes"

    assert np.all(np.isfinite(marginal)), f"NaN or inf in marginal in discrete bayes"

    return marginal, conditional
