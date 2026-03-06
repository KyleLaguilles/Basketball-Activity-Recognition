# Hang-Time HAR: Setup + Modifications Notes (Kyle)

## 1. Dataset Issues
- Original code expected flat meta structure
- Our meta.txt nested under "eu" and "us", but raw filenames use suffix "_eu" and "_na"
  - "_na" corresponds to "us" in meta.txt
- When loading the processed dataset CSV files, the subject column contained mixed data types (int and str). This occurred because:
  - Some subject IDs are purely numeric (e.g., 846)
  - Others contain alphanumeric characters (e.g., 05d8)
  - Pandas infers column types dynamically, resulting in a mixed-type column
- My Jupyter workspace didn't have a GPU node (I was having trouble getting it to set up at the time)
  - In model/train.py, the following code is always executed: loss.weight = all_class_weights.cuda()
  - I modified the code in train.py (specified in 2.), but if you're using an environment with a GPU, 
    you can replace the original lines back

## 2. Code Modifications
- Modified meta.txt 
  - changed "na" to "us"
  - added missing commas
- Modified files in data_processing folder:
  - Parse subject id and location from filename
    - Replaced lines 38-52 in data_creation.py with lines 55-87
  - Map "_na" → "us"
    - lines 65 & 66 in data_creation.py previously specified "na"; replaced with "us"
  - Keep subject id stable across sessions
  - Cast subject IDs to string before label encoding to prevent mixed dtype (int/str) errors in LabelEncoder
    - Replaced lines 42-44 in preprocess_data.py with lines 47-56 
- Modified model/train.py:
  - replaced unconditional .cuda() with conditional
    - lines 282 and 298 replaced with 277-280 and 293-296, respectively
    - might not be needed if using GPU core so may have to replace back
- Fixed mixed dtype subject IDs in data loading:
  - Added dtype={3: str} to all three pd.read_csv calls in preprocess_data.py
    to force pandas to read subject column as strings, preventing int/str 
    type inference issues
  - Added str(sbj_id) cast in data_creation.py line 57 to ensure subject IDs
    are written as strings to CSV
- Added composite confusion matrix and per-subject logging in model/validation.py:
  - Added save_composite_confusion_matrix() helper function using seaborn
    for paper-style heatmap visualization
  - Added per-subject best epoch and macro F1 tracking across LOSO folds
  - Results saved to subject_best_summary.csv and logged to W&B
- Modified model/train.py:
  - Added best_epoch tracking alongside best_metric
  - best_epoch returned from train() for per-subject logging

## 3. Environment Setup Recommended by Code Authors (CSU Tide)
- Python 3.10 required
- scikit-learn installed via conda-forge
- PyTorch installed with pytorch-cuda=11.7
- recommended commands to install:
  - conda create -n hangtime_har python=3.10
  - conda activate hangtime_har
  - conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
  - pip install -r requirements.txt
- original requirements.txt:
  - neptune==1.1.1 
  - scikit-learn==1.2.2 
  - matplotlib==3.7.1
  - prettytable==3.5.0

## 4. How I Setup my Environment:
- created 'data' folder in 'hangtime_har' code folder; pasted hangtime_har dataset inside & renamed to 'raw'
- commands:
  - conda create -n hangtime_har python=3.11
  - conda activate hangtime_har
  - conda install pytorch pytorch-cuda -c pytorch -c nvidia
  - pip install -r requirements.txt 
  - python src/data_processing/data_creation.py
    - creates files hangtime_drill_data.csv, hangtime_game_data.csv, and hangtime_warmup_data.csv inside 'data' folder
  - python src/main.py --gpu cpu --epochs 1 --batch_size 64 --wandb
    - example parameters; just wanted to get it to run without  much downtime
  - python src/main.py  --gpu cuda:0 --seed 1 --epochs 30 --batch_size 100 --wandb
    - after reserving GPU on CSU TIDE and migrating to WandB
- also added the raw and preprocessed CSV files to .gitignore so that large files wouldn't be pushed

## 5. W & B Migration Notes
- removed Neptune dependency
  - removed all Neptune-specific logging
  - replaced with Weights & Biases (wandb)
- running with W & B logging enabled:
  - python main.py --gpu cpu --epochs 1 --batch_size 64 --wandb
  - must login once per environment:
    - wandb login
- during LOSO cross-validation:
  - originally used wandb.log(..., step=e)
  - caused warning: Tried to log to step 0 that is less than the current step...
  - epoch counter resets per subject; W & B requires montonically increasing steps without a run
  - Fix: removed explicit step=e; W & B auto-increments step internally
- replaced Neptune-style logging with: wandb.log({...})
  - fixed best-model handling:
    - best epoch checkpoint is now properly stored & restored
    - improved prediction accumulation
      - avoid repeated np.concatenate
    - conditional pin_memory depending on CUDA availability
- removed leftover File(...) Neptune artifact usage
  - replaced confusion matrix uploads with:
    - wandb.Image(...)