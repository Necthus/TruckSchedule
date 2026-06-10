$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$log_file = Join-Path $script_dir "train.log"
$project_dir = Split-Path -Parent (Split-Path -Parent $script_dir)

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

python "$project_dir\main_process.py" --reposition_train_mode --reposition_method ScratchCostOnlyRL --reposition_episode_num 240 --save_model_frequency 20 --validation_frequency 20 --early_stop_patience 5 --early_stop_min_delta 0.0 --rl_lr 0.0005 *> "$log_file"
