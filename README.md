# CycleGAN-TensorFlow
CycleGAN implementation in tensorflow using keras layers

Supports up to 2 GPUs

## Features
- Supports up to 2 GPUs
- Outputs a TensorBoard log file and the model checkpoints in the 'runs' directory
- The model checkpoints are at the Keras hdf5 saved model format. To load a model, use keras.models.load_model(path_to_model)

## Configuration
See example_config.py

## Usage 
### Training the models from scratch
```shell
python train.py config_file dataA_path dataB_path
```

### Resuming training
```shell
python src/train.py config_file dataA_path dataB_path path_to_checkpoints epoch
```
### Generating samples
```shell
python src/generate.py checkpoint_path files_path output_path
```

## Dependencies
- python3
- numpy
- tensorflow >= 1.10
- python-opencv
- progressbar2

## Models
- Vanilla CycleGAN generator and discriminator
- PatchGAN discriminator
- UNet generator
