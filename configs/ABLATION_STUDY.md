# Ablation Study Configs for Coffee Pod Pickup Task

**Dataset:** `dageorge1111/v160_coffee_pod_train`
**Task:** Pick up a coffee pod from the table
**Demos:** ~150 runs
**Observations:** Wrist + ego camera images, joint angles
**Actions:** Joint velocity trajectories

---

## Baseline Configs

| Config | Policy | Description |
|--------|--------|-------------|
| `v160_act_pans.yaml` | ACT | Baseline ACT config |
| `v160_dp_pans.yaml` | Diffusion | Baseline Diffusion Policy config |

---

## ACT Ablations

| Config | Change | Value | Baseline | Reasoning |
|--------|--------|-------|----------|-----------|
| `v161_act_pans.yaml` | `image_transforms.enable` | `false` | `true` | Test if augmentation helps or hurts |
| `v162_act_pans.yaml` | `chunk_size`, `n_action_steps` | `15` | `100` | Very short chunks - more reactive to pod location |
| `v163_act_pans.yaml` | `n_obs_steps` | `2` | `1` | Motion context helps for reaching |
| `v164_act_pans.yaml` | `kl_weight` | `1.0` | `10.0` | Lower KL - there's only one way to pick up a pod |
| `v165_act_pans.yaml` | `lr`, `peak_lr` | `5.0e-5` | `1.0e-5` | Higher LR - simple task can learn faster |
| `v166_act_pans.yaml` | `chunk_size`, `n_action_steps` | `30` | `100` | Shorter chunks - task is quick, 100 is overkill |

### ACT Key Parameters Explained

- **`chunk_size`**: How many future actions the model predicts at once. Shorter = more reactive, longer = smoother.
- **`n_action_steps`**: How many of those predicted actions to actually execute before re-predicting.
- **`n_obs_steps`**: How many past frames to use as input. More = better motion understanding.
- **`kl_weight`**: VAE regularization. Higher = more consistent actions, lower = more diverse.
- **`lr` / `peak_lr`**: Learning rate. Higher = faster learning but risk instability.

---

## Diffusion Policy Ablations

| Config | Change | Value | Baseline | Reasoning |
|--------|--------|-------|----------|-----------|
| `v161_dp_pans.yaml` | `contrast.weight`, `saturation.weight` | `0.1`, `0.5` | `1.0`, `1.0` | Reduced augmentation intensity |
| `v162_dp_pans.yaml` | `horizon`, `n_action_steps` | `32`, `16` | `16`, `8` | Longer horizon - might give smoother reach |
| `v163_dp_pans.yaml` | `n_obs_steps` | `4` | `2` | More visual history for tracking approach |
| `v164_dp_pans.yaml` | `vision_backbone` | `resnet18` | `resnet34` | Smaller backbone - less overfitting with 150 demos |
| `v165_dp_pans.yaml` | `crop_shape` | `[160, 160]` | `[200, 200]` | Tighter crops - force focus on relevant area |
| `v166_dp_pans.yaml` | `horizon`, `n_action_steps` | `8`, `4` | `16`, `8` | Shorter horizon - more reactive for quick task |

### Diffusion Policy Key Parameters Explained

- **`horizon`**: Total action sequence length the model predicts.
- **`n_action_steps`**: How many actions to execute before re-predicting.
- **`n_obs_steps`**: Number of past observation frames.
- **`vision_backbone`**: CNN for image encoding. resnet18 < resnet34 in capacity.
- **`crop_shape`**: Random crop size for augmentation. Smaller = more variety.
- **`num_train_timesteps`**: Diffusion denoising steps during training.

---

## Recommendations for Coffee Pod Task

Given the task is a simple, quick pick-and-place:

1. **Shorter action chunks** (15-30 for ACT, 8-16 horizon for DP) likely work better than long predictions
2. **Observation history** (`n_obs_steps > 1`) helps since you're predicting velocities
3. **Smaller models** might generalize better with only 150 demos
4. **ACT might outperform DP** - simpler task doesn't need diffusion's expressiveness

---

## How to Run

```bash
# ACT training
python lerobot/scripts/train.py --config configs/v162_act_pans.yaml

# Diffusion Policy training
python lerobot/scripts/train.py --config configs/v166_dp_pans.yaml
```

---

## Tracking Results

| Config | Train Loss | Eval Success Rate | Notes |
|--------|------------|-------------------|-------|
| v160_act_pans | | | Baseline |
| v160_dp_pans | | | Baseline |
| v161_act_pans | | | No augmentation |
| v162_act_pans | | | chunk=15 |
| v163_act_pans | | | n_obs=2 |
| v164_act_pans | | | kl=1.0 |
| v165_act_pans | | | lr=5e-5 |
| v166_act_pans | | | chunk=30 |
| v161_dp_pans | | | Reduced aug |
| v162_dp_pans | | | horizon=32 |
| v163_dp_pans | | | n_obs=4 |
| v164_dp_pans | | | resnet18 |
| v165_dp_pans | | | crop=160 |
| v166_dp_pans | | | horizon=8 |
