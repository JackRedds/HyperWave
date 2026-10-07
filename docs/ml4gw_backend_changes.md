# Changes to the ml4gw waveform backend: burst waveforms

This note records how the ml4gw waveform backend was extended from CBC-only to also
generate burst waveforms, which waveforms were added, and the open problems
in the current code.

- **Baseline:** `src/hyperwave/` as of `8af6f64` (the last upstream commit before
  these changes). At that point the ml4gw backend supported only `IMRPhenomD`,
  `IMRPhenomPv2` and `TaylorF2`.
- **Changes covered:** `370e562` ("adding bursts to ml4gw backend", 2026-07-24) and
  `2bdd62c` ("Updates to ml4gw backend", 2026-09-27).

To see the full diff:
`git diff -M 8af6f64 HEAD -- src/hyperwave/ml4gw.py src/hyperwave/detectors/waveforms/`

---

## 1. Waveforms added

All of these come from `ml4gw.waveforms`, run in the time domain, and are
FFT'd onto the analysis grid by HyperWave. All are selected with
`waveform_backend="ml4gw"`.

| `approximant` | Waveform | Parameters (passed straight to ml4gw) | Constructor |
|---|---|---|---|
| `SineGaussian` | Sine-Gaussian | `quality`, `frequency`, `hrss`, `phase`, `eccentricity`, `shifts` | `SineGaussian(sample_rate, duration)` |
| `WhiteNoiseBurst` | Band-limited white-noise burst (LAL-style: Gaussian time + frequency envelopes, Tukey window) | `frequency`, `bandwidth`, `eccentricity`, `phase`, `int_hdot_squared`, `duration` | `WhiteNoiseBurst(sample_rate, duration)` |
| `CosmicString` | Cosmic-string cusp/kink | `power`, `amplitude`, `f_high` | `CosmicString(sample_rate, duration)` |
| `Gaussian` | Gaussian pulse | `hrss`, `polarization`, `eccentricity`, `duration` | `Gaussian(sample_rate, duration)` |
| `MultiWaveform` | Sum of several waveforms (renamed from `MultiSineGaussian` in `2bdd62c`) | treated as the sine-Gaussian set (see 4.3) | `MultiWaveform(generator, sample_rate, duration, norm=True)` |

The extrinsic parameters `ra`, `dec`, `psi` and `geocent_time` are handled by
HyperWave's `Template` projection as before, so each burst approximant is called
with its intrinsic parameters only. Example (from `scripts/wavelet_reconstruction.py`):

```python
injector = GW(noise, approximant="WhiteNoiseBurst",
              parameters=["frequency", "bandwidth", "eccentricity", "phase",
                          "int_hdot_squared", "psi", "ra", "dec", "duration"],
              static_parameters={"geocent_time": trigger_time},
              waveform_backend="ml4gw")
```

Example injection values and bilby priors for each family are in
`scripts/wavelet_reconstruction.py` (`make_sg`, `make_cs`, ...) and in
`scripts/wavelet_injection_campaign.py` (`_sg_priors`, `_cs_priors`, ...).

### 1.1 These waveforms come from a fork of ml4gw

`CosmicString`, `Gaussian`, `WhiteNoiseBurst`, `MultiWaveform` (and a
`MorletGabor`) are in **`JackRedds/ml4gw`**. The venv has commit `648e14e` installed
from `ssh://git@github.com/JackRedds/ml4gw.git`, reporting version `0.0.1`.
`pyproject.toml` still requires `ml4gw>=0.7.12` from PyPI. The fork URL is there
only as a commented-out alternative:

```toml
ml4gw = ["ml4gw>=0.7.12; python_version < '3.13'"]
# ml4gw = [
#     "ml4gw @ git+ssh://git@github.com:JackRedds/ml4gw.git"
# ]
```

So `pip install hyperwave[ml4gw]` gets PyPI ml4gw. If that release lacks these
classes, `require_ml4gw_modules()` raises `ImportError` for **every** ml4gw use,
including the CBC approximants, because it imports all of them in one statement.
The PyPI version wasn't checked here, and the fork reports version `0.0.1`, which
doesn't satisfy `>=0.7.12`.

---

## 2. Structural changes

### 2.1 Backends moved into a `backends/` subpackage

```
detectors/waveforms/base.py           -> detectors/waveforms/backends/base.py
detectors/waveforms/lal_backend.py    -> detectors/waveforms/backends/lal_backend.py
detectors/waveforms/ml4gw_backend.py  -> detectors/waveforms/backends/ml4gw_backend.py
```

`detectors/waveforms/__init__.py` imports from the new paths. The package-level
`ML4GWWaveform` name is still loaded lazily, so torch/ml4gw are only needed when
it's used.

### 2.2 `LALWaveform` renamed to `LALCBCWaveform`

The name now says that this class only generates CBC waveforms.
`detectors.waveforms` exports `LALCBCWaveform` instead of `LALWaveform`.

### 2.3 ml4gw backend split into a base class and two subclasses

`backends/ml4gw_backend.py`:

- **`ML4GWWaveform`** (base). Has a table from approximant name to ml4gw class and to
  that waveform's parameter set. It owns the shared `polarizations()` pipeline:
  normalise the batch → `parameter_adapter(batch)` → generator → `rfft / fs` →
  `_fft_correction` → `_phase_correction` → copy onto the analysis frequency grid.
  Subclasses customise it through two hooks, `parameter_adapter` and
  `_phase_correction`.
- **`ML4GWCBCWaveform`**. Builds the approximant module, wraps it in
  `TimeDomainCBCWaveformGenerator`, converts HyperWave parameters to bilby
  conventions with `HyperwaveToCBC.convert`, and keeps the CBC-specific corrections
  from before: the coalescence time reference and the empirical
  `phase → π − 2·phase` fix that matches LAL.
- **`ML4GWBurstWaveform`**. Builds the ml4gw burst module directly
  (`cls(sample_rate, duration)`). Its `parameter_adapter` passes the batch through
  unchanged as float64 tensors. There is no phase correction (the identity).

`template.py` has a new `BACKENDS` table that chooses which subclass to use for each
approximant. Before, it always built `ML4GWWaveform`.

### 2.4 Parameter sets for each waveform family

In `backends/base.py`:

- `INTRINSIC_PARAMETERS` was renamed to `INTRINSIC_PARAMETERS_CBC`.
- New sets were added: `INTRINSIC_PARAMETERS_SG`, `_CS`, `_WNB`, `_GAUSSIAN` and
  `EXTRINSIC_DEFAULTS` (`psi`, `dec`, `ra`, `geocent_time`).
- `normalize_intrinsic_batch(params, n, intrinsic_params)` now takes the parameter
  set to normalise against. Before, it always used the CBC set. Burst parameters
  have no defaults, so a missing one raises `KeyError`.

### 2.5 CBC parameter conversion moved out of `Template`

`Template._to_intrinsic` (chirp mass/mass ratio → component masses,
`cos_theta_jn` → `theta_jn`, signed spin + `cos_tilt` → `(a, tilt)`) became
`HyperwaveToCBC.convert` in the new `detectors/waveforms/parameters.py`, together
with `component_masses`. `Template.make_injections_to_ifo_batch` now passes the
named HyperWave parameters straight to `backend.polarizations()`. Each backend
converts them itself: the CBC backends (LAL and ml4gw) call `HyperwaveToCBC`, and
the burst backend uses the names as they are.

### 2.6 `hyperwave/ml4gw.py`

`ML4GWModules` and `require_ml4gw_modules()` now also load `SineGaussian`,
`MultiWaveform`, `WhiteNoiseBurst`, `CosmicString` and `Gaussian` from
`ml4gw.waveforms`.

### 2.7 Seeding the white-noise burst

ml4gw's `WhiteNoiseBurst` draws its noise with `torch.randn` and doesn't take a seed.
The scripts call `torch.manual_seed(seed)` before every ml4gw generation (see
`build_problem` in `wavelet_reconstruction.py`, `pure_signal` in
`wavelet_injection_campaign.py`). This keeps the injected data, the "pure signal"
used for the optimal SNR, and the CPU twin used by the GPU check identical. Anything
else that generates a `WhiteNoiseBurst` must do the same, or each call returns a
different waveform.

### 2.8 Other changes in these commits

- New `skymap` extra (`healpy>=1.16`) in `pyproject.toml`. It belongs to the sky-map work,
  not the backend.
- `notebooks/` was added to `.gitignore`.

---

## 3. Status

What has been run:

- All five families inject and give finite network SNRs through
  `GW(..., waveform_backend="ml4gw")` on CPU (SNRs for the default injection values
  in `wavelet_reconstruction.py`: SG 80.4, CS 79.4, Gaussian 56.9, WNB 22.6).
- A short end-to-end reconstruction of a cosmic string with sky sampling runs.
- There are no unit tests for the burst backend yet (`tests/test_waveform_backends.py`
  only covers the LAL CBC backend), and nothing compares the burst waveforms
  against LAL's burst waveforms.

---

## 4. Open problems in the current code

These were found while writing this note. None of them is fixed yet. 4.1–4.4 were
confirmed by running the code; 4.5 is worked out from reading the code.

### 4.1 Broken `__all__` in `ml4gw_backend.py`

```python
__all__ = ["ML4GWWaveform", "ML4GWCBCWWaveform", "ML4GWBurstWaveform",
           "ML4GW_APPROXIMANTS", "component_masses"]
```

`ML4GWCBCWWaveform` is a typo (extra `W`), and `component_masses` isn't defined in
this module. Running `from ...backends.ml4gw_backend import *` raises
`AttributeError`. Normal named imports still work.

### 4.2 The LAL backend test fails to import

`tests/test_waveform_backends.py` imports `LALWaveform` from
`backends.lal_backend`, but the class is now `LALCBCWaveform`, so the test fails
with `ImportError`. The README (line 117:
`from hyperwave.detectors.waveforms import LALWaveform, ML4GWWaveform`) and the
`lal_backend.py` module docstring still use the old name too.

### 4.3 `MultiWaveform` / `MultiSineGaussian` mismatch

`2bdd62c` renamed `MultiSineGaussian` to `MultiWaveform` in `ml4gw.py` and
`ml4gw_backend.py`, but **not** in `template.py`'s `BACKENDS` table. So:

- `approximant="MultiWaveform"` → `KeyError` in `Template._build_backend`;
- `approximant="MultiSineGaussian"` → finds the burst class, then hits a `KeyError` in
  `ML4GWWaveform.__init__`.

`MultiWaveform` can't be used either way. Even with the name fixed, its
constructor is `MultiWaveform(generator, sample_rate, duration, norm)`, but
`ML4GWBurstWaveform` calls `cls(sample_rate, duration)`. It is also given the
sine-Gaussian parameter set, while its `forward(**parameters)` takes whatever the
wrapped generator needs. It needs its own subclass.

### 4.4 Debug `print` in the CBC path

`ML4GWCBCWaveform.polarizations` contains `print(intrinsic)`. This prints the whole
parameter dict on every call, which floods the output for likelihood evaluations and
slows down batched runs.

### 4.5 Burst time placement only lines up for the default segment layout

`_fft_correction` (which used to be CBC-only) now runs for bursts as well. It
multiplies by `exp(+2πi f t_c)` with `t_c = duration − right_pad`, which is the
trigger's offset from the start of the segment, to move the signal from `t_c` to
`t = 0`. ml4gw's CBC generator places the merger at `t_c`, so this is correct for
CBCs. The burst generators centre the signal at **`duration / 2`** instead (they
build `times -= duration / 2`).

The two agree only when the trigger is in the middle of the segment. With
`DetectorNoise`'s default `post_trigger_duration=2` and the scripts'
`--duration 4`, that holds (t_c = 2 = 4/2). With other durations, e.g.
`--duration 8` (t_c = 6, centre = 4), a burst would be injected about
`duration/2 − post_trigger_duration` away from `geocent_time`. A fix is to
override `_fft_correction` in `ML4GWBurstWaveform` to shift by `duration / 2`
(and add a test that the burst peaks at `geocent_time`).

### 4.6 Smaller code-quality notes

- `ML4GWWaveform.parameter_adapter` does `return NotImplementedError` rather than
  `raise`, so a subclass that forgets to override it fails later with a confusing error.
- The burst parameter sets are Python `set`s, while the CBC set is a tuple. This works
  because the adapters pass the values by keyword, but the order isn't fixed.
- `EXTRINSIC_DEFAULTS` is a set of names, not a dict of defaults, and nothing uses it yet.
- A `# Need to fix this` comment was left in `normalize_intrinsic_batch`.
