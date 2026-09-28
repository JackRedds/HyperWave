"""HyperWave: hyperbolic likelihood tools for gravitational-wave data analysis."""

from importlib import metadata

__version__ = "2.0.0"

try:  # Prefer distribution version if installed
    __version__ = metadata.version("hyperwave")
except metadata.PackageNotFoundError:
    pass

from . import detectors, plots
from .detectors import (
    Detector,
    Interferometer,
    InterferometerList,
    PowerSpectralDensity,
    StrainData,
    Template,
    WaveletTemplate,
)
from .detectors.lvk import GW, DetectorNoise
from .inference import (
    AdaptiveFlowProposal,
    ContextAwareBirthRJMove,
    DataInference,
    FlowTrainingCallback,
    InferenceRunner,
    LVKinference,
    SNRPrior,
    build_wavelet_priors,
    flow_backend_available,
    make_flow_distribution_move,
    make_flow_rj_move,
)
from .likelihoods import (
    GWLikelihoods,
    HeterodyneLikelihood,
    LogLike,
    WaveletLikelihood,
    gpu_backend_available,
    loglike,
)
from .ml4gw import ml4gw_available, torch_cuda_available
from .result import Result
from .skymap import (
    credible_area,
    crossmatch_sky,
    healpix_skymap,
    load_sky_samples,
    searched_area,
    searched_probability,
    sky_localization_summary,
    thin_sky_samples,
)
from .utils import load_object, save_object
from . import validation

__all__ = [
    "__version__",
    # io / results
    "load_object",
    "save_object",
    "Result",
    # sky localization
    "load_sky_samples",
    "searched_area",
    "searched_probability",
    "credible_area",
    "sky_localization_summary",
    "thin_sky_samples",
    "healpix_skymap",
    "crossmatch_sky",
    # inference (bilby priors retained here)
    "LVKinference",
    "InferenceRunner",
    "DataInference",
    "AdaptiveFlowProposal",
    "ContextAwareBirthRJMove",
    "FlowTrainingCallback",
    "flow_backend_available",
    "make_flow_distribution_move",
    "make_flow_rj_move",
    # likelihoods
    "GWLikelihoods",
    "HeterodyneLikelihood",
    "WaveletLikelihood",
    "LogLike",
    "loglike",
    "gpu_backend_available",
    # detectors / waveforms (bilby-free)
    "Detector",
    "PowerSpectralDensity",
    "StrainData",
    "Interferometer",
    "InterferometerList",
    "Template",
    "WaveletTemplate",
    "DetectorNoise",
    "GW",
    # wavelet reconstruction
    "build_wavelet_priors",
    "SNRPrior",
    # acceleration probes
    "ml4gw_available",
    "torch_cuda_available",
    # subpackages
    "detectors",
    "plots",
    "validation",
]
