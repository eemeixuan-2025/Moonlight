# Run Streamlit app with recommended virtualenv and port
# Usage: Right-click -> Run with PowerShell, or run from PowerShell prompt

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root
if (-Not (Test-Path '.venv')) {
    python -m venv .venv
}
Write-Host 'Activating virtual environment...'
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host 'Starting Streamlit (if Windows firewall prompts please Allow access)...'
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
