# Bit depth Expansion with transformer
## Inference 
To run this project on the inference script, simply use the following commands
```
cd KAIR
python BDEinference_script.py
```
This will provide help for running the script
```
usage: BDEinference_script.py [-h] --input_folder INPUT_FOLDER [--expected_folder EXPECTED_FOLDER]
                              [--output_folder OUTPUT_FOLDER] --model_path MODEL_PATH [--device DEVICE]
                              [--framework FRAMEWORK]
BDEinference_script.py: error: the following arguments are required: --input_folder, --model_path
```
Put in expected folder if you have an existing folder of the real images the 3 bit images were extracted from and the script also compare metrics such as MSE and PSNR.
Folder `KAIR/models/networks/network_bit_depth_expansion_Ultra.py` contains the code for the model if needed for training.
SwinIR repository is included for reference since it is a part of our base model.
## Training
For training code, refer to the instructions at KAIR for training [SWINIR](https://github.com/cszn/KAIR/blob/master/docs/README_SwinIR.md) and use `KAIR/options/swinir/train_bit_depth_expansion.json`
