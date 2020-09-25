"""

"""
# %% Imports

# types
from typing import (
    Tuple,
    List,
    TypeVar,
    Optional,
    Dict,
    Any,
    Union,
    Sequence,
    Generic,
    Iterable,
)
from mixturedata import MixtureParameters
from gaussparams import GaussParams
from estimatorduck import StateEstimator

# packages
from dataclasses import dataclass
# from singledispatchmethod import singledispatchmethod
import numpy as np
from scipy import linalg
from scipy.special import logsumexp

# local
import discretebayes
import mixturereduction

# %% TypeVar and aliases
MT = TypeVar("MT")  # a type variable to be the mode type

# %% IMM
@dataclass
class IMM(Generic[MT]):
    # The M filters the IMM relies on
    filters: List[StateEstimator[MT]]
    # the transition matrix. PI[i, j] = probability of going from model i to j: shape (M, M)
    PI: np.ndarray
    # init mode probabilities if none is given
    initial_mode_probabilities: Optional[np.ndarray] = None

    def __post_init__(self):
        # This have to be satisfied!
        if not np.allclose(self.PI.sum(axis=1), 1):
            raise ValueError("The rows of the transition matrix PI must sum to 1.")

        # Nice to have a reasonable initial mode probability
        if self.initial_mode_probabilities is None:
            eigvals, eigvecs = linalg.eig(self.PI)
            self.initial_mode_probabilities = eigvecs[:, eigvals.argmax()]
            self.initial_mode_probabilities = (
                self.initial_mode_probabilities / self.initial_mode_probabilities.sum()
            )

    def mix_probabilities(
        self,
        immstate: MixtureParameters[MT],
        # sampling time
        Ts: float,
    ) -> Tuple[
        np.ndarray, np.ndarray
    ]:  # predicted_mode_probabilities, mix_probabilities: shapes = ((M, (M ,M))).
        # mix_probabilities[s] is the mixture weights for mode s
        """Calculate the predicted mode probability and the mixing probabilities."""
        
        
        predicted_mode_probabilities, mix_probabilities = discretebayes.discrete_bayes(immstate.weights, self.PI) # TODO hint: discretebayes.discrete_bayes

        # Optional assertions for debugging
        assert np.all(np.isfinite(predicted_mode_probabilities))
        assert np.all(np.isfinite(mix_probabilities))
        assert np.allclose(mix_probabilities.sum(axis=1), 1)
        
        return predicted_mode_probabilities, mix_probabilities

    def mix_states(
        self,
        immstate: MixtureParameters[MT],
        # the mixing probabilities: shape=(M, M)
        mix_probabilities: np.ndarray,
    ) -> List[MT]:
        
        means = []
        covs = []
        
        for component in immstate.components:
            means.append(component.mean)
            covs.append(component.cov)
        
        # now we have a list of all the means and covariances
        # to mix it we can use eq. 6.29 and 6.30 from the book
        
        mixed_states = []
        
        for weights in mix_probabilities:
            mean_bar, cov_bar = mixturereduction.gaussian_mixture_moments(weights, means, covs)
            mixed_states.append(GaussParams(mean_bar, cov_bar))
        
        mixed_states = np.array(mixed_states)
        
        # mixed_states = self.reduce_mixture(immstate)
        
        print_mixed_states = False
        if print_mixed_states:
            print("\n\nmixed_states:")
            print("mix_probabilities", mix_probabilities)
            print("means", means)
            print("covs", covs)
            print("mixed_states", mixed_states, np.shape(mixed_states))
            
        
        return mixed_states

    def mode_matched_prediction(
        self,
        mode_states: List[MT],
        # The sampling time
        Ts: float,
    ) -> List[MT]:
        
        
        modestates_pred = [] # TODO
        
        for filt, mode_state in zip(self.filters, mode_states):
            pred_state = filt.predict(mode_state, Ts)
            modestates_pred.append(pred_state)
        
        
        modestates_pred = np.array(modestates_pred)
        
        print_mode_matched_prediction = False
        if print_mode_matched_prediction:
            print("\n\nmode_matched_prediction:")
            print("mode_states", mode_states, np.shape(mode_states))
            print("self.filters", self.filters, np.shape(self.filters))
        
        return modestates_pred

    def predict(
        self,
        immstate: MixtureParameters[MT],
        # sampling time
        Ts: float,
    ) -> MixtureParameters[MT]:
        """
        Predict the immstate Ts time units ahead approximating the mixture step.

        Ie. Predict mode probabilities, condition states on predicted mode,
        appoximate resulting state distribution as Gaussian for each mode, then predict each mode.
        """
        
        # TODO: proposed structure
        predicted_mode_probability, mixing_probability = self.mix_probabilities(immstate, Ts) # TODO

        mixed_mode_states: List[MT] = self.mix_states(immstate, mixing_probability) # TODO

        predicted_mode_states = self.mode_matched_prediction(mixed_mode_states, Ts) # TODO

        predicted_immstate = MixtureParameters(
            predicted_mode_probability, predicted_mode_states
        )
        
        print_predict = False
        if print_predict:    
            print("\n\npredict:")
            print("predicted_mode_probability", predicted_mode_probability, np.shape(predicted_mode_probability))
            print("mixing_probability", mixing_probability, np.shape(mixing_probability))
            print("mixed_mode_states", mixed_mode_states, np.shape(mixed_mode_states))
            print("predicted_mode_states", predicted_mode_states, np.shape(predicted_mode_states))
            print("predicted_immstate", predicted_immstate)
        
        return predicted_immstate

    def mode_matched_update(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> List[MT]:
        """Update each mode in immstate with z in sensor_state."""

        updated_state = [] # TODO

        for filt, mode_state in zip(self.filters, immstate.components):
            upd_state = filt.update(z, mode_state, sensor_state)
            updated_state.append(upd_state)

        return np.array(updated_state)

    def update_mode_probabilities(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Dict[str, Any] = None,
    ) -> np.ndarray:
        """Calculate the mode probabilities in immstate updated with z in sensor_state"""

        mode_loglikelihood = [] # TODO

        # print("update_mode_probabilities:")
        # print("immstate.components", immstate.components, np.shape(immstate.components))

        for filt, comp in zip(self.filters, immstate.components):
            # print(type(comp))
            log_lhood = filt.loglikelihood(z, comp, sensor_state)
            mode_loglikelihood.append(log_lhood)
            
        
        mode_loglikelihood = np.array(mode_loglikelihood)
        
        pred_mode_probs = immstate.weights
        
        # print(mode_loglikelihood, pred_mode_probs)
        
        normalization = np.sum(np.exp(mode_loglikelihood) * pred_mode_probs)
        
        log_pred_mode_probs = np.log(pred_mode_probs)
        log_normalization = np.log(normalization)
        
        # eq. 6.33 in book
        log_updated_mode_probs = mode_loglikelihood + log_pred_mode_probs - log_normalization

        updated_mode_probabilities = np.exp(log_updated_mode_probs) # TODO

        # Optional debuging
        assert np.all(np.isfinite(updated_mode_probabilities))
        assert np.allclose(np.sum(updated_mode_probabilities), 1)

        return updated_mode_probabilities

    def update(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Dict[str, Any] = None,
    ) -> MixtureParameters[MT]:
        """Update the immstate with z in sensor_state."""
        
        
        updated_weights = self.update_mode_probabilities(z, immstate, sensor_state)# TODO
        updated_states = self.mode_matched_update(z, immstate, sensor_state) # TODO

        updated_immstate = MixtureParameters(updated_weights, updated_states)
        
        print_update = False
        if print_update:
            print("update:")
            print("immstate", immstate)
        
        return updated_immstate

    def step(
        self,
        z,
        immstate: MixtureParameters[MT],
        Ts: float,
        sensor_state: Dict[str, Any] = None,
    ) -> MixtureParameters[MT]:
        """Predict immstate with Ts time units followed by updating it with z in sensor_state"""

        predicted_immstate = self.predict(immstate, Ts) # TODO
        updated_immstate = self.update(z, immstate, sensor_state) # TODO

        return updated_immstate

    def loglikelihood(
        self,
        z: np.ndarray,
        immstate: MixtureParameters,
        *,
        sensor_state: Dict[str, Any] = None,
    ) -> float:

        # THIS IS ONLY NEEDED FOR IMM-PDA. You can therefore wait if you prefer.

        mode_conditioned_ll = None # TODO in for IMM-PDA

        ll = None # TODO

        return ll

    def reduce_mixture(
        self, immstate_mixture: MixtureParameters[MixtureParameters[MT]]
    ) -> MixtureParameters[MT]:
        """Approximate a mixture of immstates as a single immstate"""
        print("\n\nreduce_mixture:")
        print("type(immstate_mixture)", type(immstate_mixture))
        # extract probabilities as array
        weights = immstate_mixture.weights
        component_conditioned_mode_prob = np.array(
            [c.weights.ravel() for c in immstate_mixture.components]
        )

        # flip conditioning order with Bayes
        mode_prob, mode_conditioned_component_prob = discretebayes.discrete_bayes(weights, component_conditioned_mode_prob) # TODO

        # Hint list_a of lists_b to list_b of lists_a: zip(*immstate_mixture.components)
        mode_states = None # TODO:

        immstate_reduced = MixtureParameters(mode_prob, mode_states)

        return immstate_reduced

    def estimate(self, immstate: MixtureParameters[MT]) -> GaussParams:
        """Calculate a state estimate with its covariance from immstate"""

        # ! You can assume all the modes have the same reduce and estimate function
        # ! and use eg. self.filters[0] functionality
        #data_reduced = # TODO
        
        means = []
        covs = []
        
        for comp in immstate.components:
            means.append(comp.mean)
            covs.append(comp.cov)
            
        means = np.array(means)
        covs = np.array(covs)
            
        mean_reduced, cov_reduced = mixturereduction.gaussian_mixture_moments(immstate.weights, means, covs)
        
        estimate = GaussParams(mean_reduced, cov_reduced)
        
        return estimate

    def gate(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        gate_size: float,
        sensor_state: Dict[str, Any] = None,
    ) -> bool:
        """Check if z is within the gate of any mode in immstate in sensor_state"""

        # THIS IS ONLY NEEDED FOR PDA. You can wait with implementation if you want
        gated_per_mode = None # TODO

        gated = None # TODO
        return gated

    def NISes(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, np.ndarray]:
        """Calculate NIS per mode and the average"""
        NISes = np.array(
            [
                fs.NIS(z, ms, sensor_state=sensor_state)
                for fs, ms in zip(self.filters, immstate.components)
            ]
        )

        innovs = [
            fs.innovation(z, ms, sensor_state=sensor_state)
            for fs, ms in zip(self.filters, immstate.components)
        ]

        v_ave = np.average([gp.mean for gp in innovs], axis=0, weights=immstate.weights)
        S_ave = np.average([gp.cov for gp in innovs], axis=0, weights=immstate.weights)

        NIS = (v_ave * np.linalg.solve(S_ave, v_ave)).sum()
        return NIS, NISes

    def NEESes(
        self,
        immstate: MixtureParameters,
        x_true: np.ndarray,
        *,
        idx: Optional[Sequence[int]] = None,
    ):
        NEESes = np.array(
            [
                fs.NEES(ms, x_true, idx=idx)
                for fs, ms in zip(self.filters, immstate.components)
            ]
        )
        est = self.estimate(immstate)

        NEES = self.filters[0].NEES(est, x_true, idx=idx)  # HACK?
        return NEES, NEESes

    # @singledispatchmethod
    # def init_filter_state(
    #     self,
    #     init,  # Union[
    #     #     MixtureParameters, Dict[str, Any], Tuple[Sequence, Sequence], Sequence
    #     # ],
    # ) -> MixtureParameters:
    #     """
    #     Initialize the imm state to MixtureParameters.

    #     - If mode probabilities are not found they are initialized from self.initial_mode_probabilities.
    #     - If only one mode state is found, it is broadcasted to all modes.

    #     MixtureParameters: goes unaltered
    #     dict:
    #         ["weights", "probs", "probabilities", "mode_probs"]
    #             in this order can signify mode probabilities
    #         ["components", "modes"] signify the modes
    #     tuple: first element is mode probabilities and second is mode states
    #     Sequence: assumed to be only the mode states

    #     mode probabilities: array_like
    #     components:

    #     """  # TODO there are cases where MP unaltered can lead to trouble

    #     raise NotImplementedError(
    #         f"IMM do not know how to initialize a immstate from: {init}"
    #     )

    # @init_filter_state.register
    # def _(self, init: MixtureParameters[MT]) -> MixtureParameters[MT]:
    #     return init

    # @init_filter_state.register(dict)
    # def _(self, init: dict) -> MixtureParameters[MT]:
    #     # extract weights
    #     got_weights = False
    #     got_components = False
    #     for key in init:
    #         if not got_weights and key in [
    #             "weights",
    #             "probs",
    #             "probabilities",
    #             "mode_probs",
    #         ]:
    #             weights = np.asfarray([key])
    #             got_weights = True
    #         elif not got_components and key in ["components", "modes"]:
    #             components = self.init_components(init[key])
    #             got_components = True

    #     if not got_weights:
    #         weights = self.initial_mode_probabilities

    #     if not got_components:
    #         components = self.init_components(init)

    #     assert np.allclose(weights.sum(), 1), "Mode probabilities must sum to 1 for"

    #     return MixtureParameters(weights, components)

    # @init_filter_state.register(tuple)
    # def _(self, init: tuple) -> MixtureParameters[MT]:
    #     assert isinstance(init[0], Sized) and len(init[0]) == len(
    #         self.filters
    #     ), f"To initialize from tuple the first element must be of len(self.filters)={len(self.filters)}"

    #     weights = np.asfarray(init[0])
    #     components = self.init_compontents(init[1])
    #     return MixtureParameters(weights, components)

    # @init_filter_state.register(Sequence)
    # def _(self, init: Sequence) -> MixtureParameters[MT]:
    #     weights = self.initial_mode_probabilities
    #     components = self.init_components(init)
    #     return MixtureParameters(weights, components)

    # @singledispatchmethod
    # def init_components(self, init: "Union[Iterable, MT_like]") -> List[MT]:
    #     """ Make an instance or Iterable of the Mode Parameters into a list of mode parameters"""
    #     return [fs.init_filter_state(init) for fs in self.filters]

    # @init_components.register(dict)
    # def _(self, init: dict):
    #     return [fs.init_filter_state(init) for fs in self.filters]

    # @init_components.register(Iterable)
    # def _(self, init: Iterable) -> List[MT]:
    #     if isinstance(init[0], (np.ndarray, list)):
    #         return [
    #             fs.init_filter_state(init_s) for fs, init_s in zip(self.filters, init)
    #         ]
    #     else:
    #         return [fs.init_filter_state(init) for fs in self.filters]

    def estimate_sequence(
        self,
        # A sequence of measurements
        Z: Sequence[np.ndarray],
        # the initial KF state to use for either prediction or update (see start_with_prediction)
        init_immstate: MixtureParameters,
        # Time difference between Z's. If start_with_prediction: also diff before the first Z
        Ts: Union[float, Sequence[float]],
        *,
        # An optional sequence of the sensor states for when Z was recorded
        sensor_state: Optional[Iterable[Optional[Dict[str, Any]]]] = None,
        # sets if Ts should be used for predicting before the first measurement in Z
        start_with_prediction: bool = False,
    ) -> Tuple[List[MixtureParameters], List[MixtureParameters], List[GaussParams]]:
        """Create estimates for the whole time series of measurements. """

        # sequence length
        K = len(Z)

        # Create and amend the sampling array
        Ts_start_idx = int(not start_with_prediction)
        Ts_arr = np.empty(K)
        Ts_arr[Ts_start_idx:] = Ts
        # Insert a zero time prediction for no prediction equivalence
        if not start_with_prediction:
            Ts_arr[0] = 0

        # Make sure the sensor_state_list actually is a sequence
        sensor_state_seq = sensor_state or [None] * K

        init_immstate = self.init_filter_state(init_immstate)
        
        # # new init code from BB announcement
        
        # # parameters
        # init_mean = np.array([0, 0, 0, 0, 0])
        # init_cov = np.array(np.eye(5)*100)
        # a = 0.5
        # num_modes = 2 # CV and CT model
        
        # init_modes = GaussParams(init_mean, init_cov) # init_mean has shape (5,) and init_cov has shape (5, 5) #
        # init_mode_probabilities = np.array([a, 1-a]) # a in [0,1]
        # init_state = MixtureParameters(init_mode_probabilities, [init_modes] * num_modes) # can also be a list of different GaussParams per mode.

        immstate_upd = init_immstate
        
        # print("estimate_sequence:")
        # print("immstate", immstate_upd)

        immstate_pred_list = []
        immstate_upd_list = []
        estimates = []

        for z_k, Ts_k, ss_k in zip(Z, Ts_arr, sensor_state_seq):
            immstate_pred = self.predict(immstate_upd, Ts_k)
            immstate_upd = self.update(z_k, immstate_pred, sensor_state=ss_k)

            immstate_pred_list.append(immstate_pred)
            immstate_upd_list.append(immstate_upd)
            estimates.append(self.estimate(immstate_upd))

        return immstate_pred_list, immstate_upd_list, estimates
