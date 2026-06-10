# Project Instructions

- Use the active conda `base` environment for Python commands.
- In PowerShell, call `python` directly because the shell is expected to already be in `base`.
- Do not wrap Python runs in `cmd /c` in PowerShell scripts.
- Do not use `conda run` in PowerShell scripts for this project; it can capture and re-emit output with the wrong Windows code page.
- For redirected logs, set `$env:PYTHONUTF8 = "1"` and `$env:PYTHONIOENCODING = "utf-8"`, then redirect with PowerShell's `*>`.
