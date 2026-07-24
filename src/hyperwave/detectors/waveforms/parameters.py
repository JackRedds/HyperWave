import numpy as np
from .backends.base import INTRINSIC_PARAMETERS_CBC, EXTRINSIC_DEFAULTS



def component_masses(chirp_mass, mass_ratio):
    """HyperWave's chirp-mass/mass-ratio -> component masses (unchanged formula)."""
    total_mass = chirp_mass * (1 + mass_ratio) ** 1.2 / mass_ratio**0.6
    mass_1 = total_mass / (1 + mass_ratio)
    mass_2 = mass_1 * mass_ratio
    return mass_1, mass_2

def _spin_amplitude_and_tilt(chi, cos_tilt):
    """Signed aligned-ish spin (chi) + cos_tilt -> (a, tilt), matching ml4gw path."""
    chi = np.asarray(chi, dtype=float)
    tilt = np.arccos(np.clip(cos_tilt, -1.0, 1.0))
    amplitude = np.abs(chi)
    tilt = np.where(chi < 0, np.pi - tilt, tilt)
    return amplitude, tilt


class HyperwaveToCBC:
    @staticmethod
    def convert(named):
        """Map a dict of (N,) HyperWave arrays to a bilby-convention intrinsic dict."""
        get = named.get

        if "mass_1" in named and "mass_2" in named:
            mass_1 = np.asarray(named["mass_1"], float)
            mass_2 = np.asarray(named["mass_2"], float)
        else:
            mass_1, mass_2 = component_masses(
                np.asarray(named["chirp_mass"], float), np.asarray(named["mass_ratio"], float)
            )

        if "theta_jn" in named:
            theta_jn = np.asarray(named["theta_jn"], float)
        else:
            theta_jn = np.arccos(np.clip(np.asarray(get("cos_theta_jn", 1.0), float), -1.0, 1.0))

        if "a_1" in named:
            a_1 = np.abs(np.asarray(named["a_1"], float))
            tilt_1 = np.asarray(get("tilt_1", np.arccos(np.clip(get("cos_tilt_1", 1.0), -1.0, 1.0))), float)
        else:
            a_1, tilt_1 = _spin_amplitude_and_tilt(get("chi_1", 0.0), get("cos_tilt_1", 1.0))

        if "a_2" in named:
            a_2 = np.abs(np.asarray(named["a_2"], float))
            tilt_2 = np.asarray(get("tilt_2", np.arccos(np.clip(get("cos_tilt_2", 1.0), -1.0, 1.0))), float)
        else:
            a_2, tilt_2 = _spin_amplitude_and_tilt(get("chi_2", 0.0), get("cos_tilt_2", 1.0))

        n = mass_1.shape[0] if mass_1.ndim else 1
        ones = np.ones(n)
        intrinsic = {
            "mass_1": mass_1 * ones,
            "mass_2": mass_2 * ones,
            "luminosity_distance": np.asarray(named["luminosity_distance"], float) * ones,
            "theta_jn": theta_jn * ones,
            "phase": np.asarray(named["phase"], float) * ones,
            "a_1": a_1 * ones,
            "a_2": a_2 * ones,
            "tilt_1": tilt_1 * ones,
            "tilt_2": tilt_2 * ones,
            "phi_12": np.asarray(get("phi_12", 0.0), float) * ones,
            "phi_jl": np.asarray(get("phi_jl", 0.0), float) * ones,
            "lambda_1": np.asarray(get("lambda_1", 0.0), float) * ones,
            "lambda_2": np.asarray(get("lambda_2", 0.0), float) * ones,
            "eccentricity": np.asarray(get("eccentricity", 0.0), float) * ones,
        }
        return {k: intrinsic[k] for k in INTRINSIC_PARAMETERS_CBC}
    
__all__ = ["HyperwaveToCBC", "component_masses"]