# xs_choice: Observable Inputs to the Choice Network

## Original prompt
> rnn_utils.DatasetRNN.xs is a user-provided numpy array of shape (trials,sessions,n_inputs) that provides observation inputs, along with input from the latent units, to the update MLP network. xs_names is a list of names of the features in xs.
We need to add rnn_utils.DatasetRNN.xs_choice, which is a numpy array of the same shape as xs and provides a different set of observation inputs, along with input from the latent units, to the choice MLP network. Also need to add xs_choice_names, which is a list of names of the features in xs_choice.
Next, the disrnn. HkDisentangledRNN class needs several updates to integrate our new variable xs_choice.
The build_choice_bottlenecks method needs to incorporate both the inputs from latents and the input from observables (xs_choice), similar to the build_update_bottlenecks method.
The __call__ method presently accepts “observations” (the current trial observation inputs from xs to the update network). We need to add an input for “choice_observations” (the current trial observation inputs from xs_choice to the choice network). Then, under “Set up choice net inputs from new_latents,” we need to combine inputs from the choice observations and the latents for the choice network, similar to “Concatenate processed inputs for the update networks”

## Motivation

`DatasetRNN.xs` provides observation inputs to the **update** MLP network. The **choice** MLP network previously received input only from latent units. This change adds `xs_choice` — a separate set of observable inputs that feed directly into the choice network, enabling it to use observable features (e.g., stimulus information) in addition to learned latent representations.

## What changed

### `rnn_utils.py` — DatasetRNN

- **`get_all()` and `__next__()` now return dicts** (`{'xs': ..., 'ys': ...}`) instead of tuples. All callers in `train_network`, `split_dataset`, etc. were updated.
- **New `__init__` params**: `xs_choice` (numpy array, shape `[timestep, episode, feature]`) and `xs_choice_names` (list of str). `xs_choice` must match `xs` in timesteps and episodes but may have a different number of features.
- **Concatenation**: When `xs_choice` is present, `get_all()['xs']` and `next(dataset)['xs']` return `[xs | xs_choice]` concatenated along the feature axis. This is necessary because `hk.dynamic_unroll` passes a single array to `__call__` per timestep.
- **`get_all()['xs_choice']`** returns the raw (un-concatenated) `xs_choice` array when present.
- **`split_dataset()`** propagates `xs_choice` and `xs_choice_names` to both train and eval splits.

### `disrnn.py` — DisRnnConfig

New fields (all backward-compatible with defaults):
- `choice_obs_size: int = 0` — number of choice observation features
- `choice_net_obs_penalty: float = 0.0` — bottleneck penalty for choice observations
- `xs_choice_names: list[str] | None = None`

### `disrnn.py` — HkDisentangledRNN

- **`_build_choice_bottlenecks()`**: When `choice_obs_size > 0`, creates additional bottleneck params (`choice_net_obs_sigma_params`, `choice_net_obs_multipliers`) with shape `(choice_obs_size,)`.
- **`__call__(inputs, prev_latents)`**: Renamed param from `observations` to `inputs`. Splits the concatenated input: first `obs_size` features → update network observations, remaining `choice_obs_size` features → choice network observations. Choice observations pass through their own information bottleneck and are concatenated with latent inputs before feeding into `predict_targets()`.
- **`predict_targets()`**: Unchanged — it already uses `choice_net_inputs.shape[1]` dynamically, so it adapts to the wider input automatically.
- **`log_bottlenecks()` / `get_total_sigma()`**: Include `choice_net_obs` bottleneck params when present in the parameter dict.

### Test files

- **`rnn_utils_test.py`**: 5 new tests covering xs_choice creation, validation, auto-generated names, backward compat, and split propagation.
- **`disrnn_test.py`**: 6 new tests in `DisrnnWithChoiceObsTest` covering parameter shapes, output shapes, trainability, choice net input size, backward compat, and auxiliary metrics.

## Usage example

```python
import numpy as np
from disentangled_rnns.library import rnn_utils
from disentangled_rnns.library import disrnn

# Create dataset with choice observations
xs = np.random.randn(100, 50, 2).astype(np.float32)        # update inputs
ys = np.random.randint(0, 2, (100, 50, 1)).astype(np.float32)  # targets
xs_choice = np.random.randn(100, 50, 3).astype(np.float32) # choice inputs

dataset = rnn_utils.DatasetRNN(
    xs=xs, ys=ys,
    x_names=['prev_choice', 'prev_reward'],
    xs_choice=xs_choice,
    xs_choice_names=['stimulus_a', 'stimulus_b', 'stimulus_c'],
)

# Configure DisRNN with choice observations
config = disrnn.DisRnnConfig(
    obs_size=2,
    choice_obs_size=3,    # must match xs_choice feature count
    output_size=2,
    latent_size=10,
    choice_net_obs_penalty=0.1,
)

# Train
params, opt_state, losses = rnn_utils.train_network(
    make_network=lambda: disrnn.HkDisentangledRNN(config),
    training_dataset=dataset,
    validation_dataset=None,
    n_steps=1000,
    loss='penalized_categorical',
)
```

## Backward compatibility

All changes are backward-compatible:
- `xs_choice=None` (default) → no concatenation, identical behavior to before
- `choice_obs_size=0` (default) → no choice obs bottleneck params, `__call__` treats entire input as update observations
- Existing trained model checkpoints load correctly when `choice_obs_size=0`

## Known limitations / further work

1. **Subclasses not updated**: `MultisubjectDisRnn` and `HkNeuroDisentangledRNN` work unchanged with `choice_obs_size=0`, but don't support `xs_choice` yet. Each would need its own input-splitting and bottleneck logic.
2. **`dataset_list_to_multisubject`** doesn't propagate `xs_choice` — it bakes the concatenated form into the combined dataset.
3. **Plotting functions** (`plot_bottlenecks`, `plot_choice_rule`) don't visualize the new `choice_net_obs` bottleneck parameters.
4. **Tests verified** — all 11 `disrnn_test.py` tests and 16/20 `rnn_utils_test.py` tests pass. The 4 remaining `rnn_utils_test.py` failures are pre-existing issues unrelated to xs_choice: `test_dataset_rnn` and `test_split_dataset` expect random-sampling batch behavior but use `single` mode; `test_dataset_rnn_batch_size` sets `batch_size` incompatible with `single` mode; `test_step_network` hits a `np.random.randint(2**32)` int32 overflow on Windows.
5. **`run_experiment` fixed** — `AgentQ` and `AgentLeakyActorCritic` now return `(choice, choice_probs)` from `get_choice()` and accept an optional `xs` arg in `update()`, matching `AgentNetwork`'s interface. `run_experiment` now correctly calls `environment.step(attempted_choice)` instead of passing extra unsupported args.
