Project Moonlight — Streamlit Dashboard

Quick start

1. Put your trained model into this folder as `moonlight_model.pkl`. If you have a preprocessing pipeline (encoder/scaler), save it as `pipeline.pkl` in the same folder.

2. Install dependencies and run the app:

```bash
cd Project_Moonlight
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Notes

- The demo app loads `moonlight_model.pkl` and `pipeline.pkl` locally. Ensure your model expects the app’s input fields, or supply a `pipeline.pkl` that performs preprocessing and accepts the form inputs.
- If the charts folder is empty, rerun `python scripts/generate_model_based_pngs.py` to regenerate the model-derived PNGs used by the dashboard.
 
Firewall note:
- Windows may prompt to allow Python through the firewall when Streamlit starts. Click "Allow access" to enable the local web UI at http://localhost:8501.

If you want a one-click script, run `run_app.ps1` in this folder (PowerShell will create a venv, install dependencies, and start Streamlit).
