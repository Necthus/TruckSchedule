$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$log_file = Join-Path $script_dir "test.log"
$project_dir = Split-Path -Parent (Split-Path -Parent $script_dir)

cmd /c "set PYTHONIOENCODING=utf-8 && python `"$project_dir\main_process.py`" --reposition_method DispatchAwareRL > `"$log_file`" 2>&1"
