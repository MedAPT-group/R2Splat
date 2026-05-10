# Sweeps

This directory contains small W&B sweep templates for parameter searches.

## Beta ADC Quick

The quick beta sweep targets the `beta` training config and focuses on the
parameters that currently look most coupled to beta quality:

- ADC sensitivity: `strategy.absgrad`, `strategy.grow_grad2d`,
  `strategy.grow_scale3d`, `strategy.split_scales_div`
- Beta support/shape: `model.init_scale_multiplier`, `model.beta_gamma_init`
- Beta optimizer strength: `optim.lr_beta`

Start a sweep:

```bash
cd /home/chengzyi/myterminus/code/r2splat-r2beta
wandb sweep sweep/beta_adc_quick/sweep_beta_adc_quick.yaml
```

Launch agents on Slurm:

```bash
SWEEP_ID=<entity/project/sweep_id> sbatch sweep/beta_adc_quick/sweep.sh
```

The template uses `trainer.max_steps=15001` so ADC behavior is visible without
paying for full 30001-step runs. After the promising region is clear, increase
`trainer.max_steps` and narrow the sampled values.

## Beta Scale ADC

The main sweep for the 0_chest_cone_25_views Beta experiments is:

```bash
cd /home/chengzyi/myterminus/code/r2splat-r2beta
wandb sweep sweep/beta_scale_adc/sweep_beta_scale_adc.yaml
SWEEP_ID=<entity/project/sweep_id> sbatch sweep/beta_scale_adc/sweep.sh
```

This sweep keeps `strategy.absgrad=true` and focuses on scale bounds, init
scale multiplier, beta gamma, and grad2d ADC thresholds. It also samples a few
discrete epsilon values.
