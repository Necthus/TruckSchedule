$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$log_file = Join-Path $script_dir "train.log"
$project_dir = Split-Path -Parent (Split-Path -Parent $script_dir)

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

python "$project_dir\main_process.py" --reposition_train_mode --reposition_method DispatchAwareRL *> "$log_file"
