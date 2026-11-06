# Start TraceBrief on Windows: runs scripts\run_tracebrief.py with the arguments given here.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run-tracebrief.ps1 --config tracebrief.json --port 8786
#
# The launcher runs on the checkout's .venv-app Python when it exists, otherwise on the Python
# launcher (py -3) or python from PATH; it then finds the three environments and starts the server.
# No global ErrorActionPreference = Stop: PowerShell 5.1 would turn the launcher's notes on stderr
# into a terminating error whenever the output is redirected.
$root = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $PSScriptRoot "run_tracebrief.py"
$python = Join-Path $root ".venv-app\Scripts\python.exe"
if (Test-Path $python) {
    & $python $launcher @args
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $launcher @args
} else {
    & (Get-Command python -ErrorAction Stop).Source $launcher @args
}
exit $LASTEXITCODE
