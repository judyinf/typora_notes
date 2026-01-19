# ViTacGen: Robotic Pushing with Vision-to-Touch Generation

## Paper

### Related Works

Visual-Tactile Representation Learning



## Implementation 

### Pretrain expert policy                    

Pretrain visual-tactile expert policy (VT-Con) for data collection:

```bash
python tactile_gym/sb3_helpers/train_agent.py --exp_name vtcon -A mvitac_sac --features_extractor_class VisualTactileCMCL --seed 0 --learning_rate 1e-4
```

### Collect visual-tactile data pairs

Collect paired visual-tactile data with expert policy:

```bash
python tactile_gym/sb3_helpers/data_collection.py # Replace saved_model_dir with the path to your pretrained model
```

### Train vision-to-touch generation (VT-Gen)

Train VT-Gen for vision-to-touch generation:

### Train policy with vision-to-touch generation (ViTacGen)

Train visual-only ViTacGen policy:

```bash
python tactile_gym/sb3_helpers/train_agent.py --exp_name vitacgen -A mvitac_sac --features_extractor_class VisualCMCL_atten --seed 0 --learning_rate 1e-4
```

