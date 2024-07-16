# ColAIDiffusion

## Requirements
See `requirements.txt`

## Instructions
* Use `make.sh` to generate run script
* Use `make.py` to generate exp script
* Use `process.py` to process exp results
* Hyperparameters can be found in `config.yml` and `process_control()` in `module/hyper.py`

## Examples
 - Generate run script
    ```ruby
    bash make.sh
    ```
 - Generate run script
    ```ruby
    python make.py --mode base
    ```
 - Train with MNIST and diffusion-epsilon model
    ```ruby
    python train_model.py --control_name MNIST_diffusionEpsilon
    ```
 - Test with CIFAR10 and diffusion-x model
    ```ruby
    python test_model.py --control_name CIFAR10_diffusionX
    ```
 - Process exp results
    ```ruby
    python process.py
    ```

## Results
