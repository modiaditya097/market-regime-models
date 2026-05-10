# Manual Upload Guide: User Models Integration

## 📋 Summary of Changes Made

This branch (`feature/user-models-integration`) integrates your 4 models (Simple-HMM, MS-GARCH, HMM, HSMM) into the dashboard with full frontend execution support.

### ✅ What Was Added
- **4 new models** in `app_config.yaml` using `generic_model_tab`
- **`run_user_models.py`**: Configurable runner script for all 4 models
- **Frontend parameter controls**: Ticker dropdown (14 options), date ranges, states, features, txn cost
- **Run Model button** with progress bar and live log output
- **Auto-refresh** of plots/metrics after pipeline completes

### 📂 Key Files Modified
```
shiny_app/app_config.yaml               # Added 4 models with run_command
shiny_app/modules/generic_model_tab.py  # Added parameter sidebar + async runner
run_user_models.py                      # New unified runner script
outputs/simple_hmm/results.csv           # Fixed format for dashboard compatibility
```

---

## 🚀 Manual Steps to Create Branch & Upload

### 1️⃣ Fork the Repository (if not already forked)
1. Go to: https://github.com/modiaditya097/market-regime-models
2. Click **"Fork"** button (top right)
3. Your fork URL will be: `https://github.com/YOUR_USERNAME/market-regime-models`

### 2️⃣ Clone Your Fork (if starting fresh)
```bash
cd /Users/hh/Downloads
git clone https://github.com/YOUR_USERNAME/market-regime-models.git
cd market-regime-models
```

### 3️⃣ Add Original Repo as Upstream (if not already added)
```bash
git remote add upstream https://github.com/modiaditya097/market-regime-models.git
```

### 4️⃣ Create and Switch to Your Branch
```bash
git checkout -b feature/user-models-integration
```

### 5️⃣ Copy Your Model Files
Copy your original model outputs into this repo:
```bash
# From your original dd directory
cp -r /Users/hh/Downloads/dd/outputs/simple_hmm outputs/
cp -r /Users/hh/Downloads/dd/outputs/msgarch outputs/
cp -r /Users/hh/Downloads/dd/outputs/hmm outputs/
cp -r /Users/hh/Downloads/dd/outputs/hsmm outputs/
```

### 6️⃣ Copy/Recreate the Runner Script
Create `run_user_models.py` with the unified runner logic (see this repo's version).

### 7️⃣ Update Configuration Files
- Edit `shiny_app/app_config.yaml` to add your 4 models
- Edit `shiny_app/modules/generic_model_tab.py` to add parameter controls

### 8️⃣ Commit All Changes
```bash
git add .
git commit -m "feat: integrate user models with frontend pipeline execution

- Add Simple-HMM, MS-GARCH, HMM, HSMM models to dashboard
- Implement configurable runner script (run_user_models.py)
- Add parameter sidebar with ticker dropdown and controls
- Enable Run Model button with progress tracking
- Fix results.csv format compatibility"
```

### 9️⃣ Push to Your Fork
```bash
git push origin feature/user-models-integration
```

### 🔟 Create Pull Request
1. Go to your fork on GitHub
2. Click **"Compare & pull request"**
3. Title: `Feature: User Models Integration`
4. Description: Include summary of features
5. Click **"Create pull request"**

---

## 🧪 Testing Before Upload

### Verify Dashboard Works
```bash
cd market-regime-models
python3 -m shiny run shiny_app/app.py --port 8008
```
Then visit http://127.0.0.1:8008 and test:
- All 4 model tabs appear
- Parameter sidebar shows ticker dropdown
- "Run Model" button executes with progress bar
- Plots/metrics update after run completes

### Verify Runner Script
```bash
python3 run_user_models.py --model simple_hmm
python3 run_user_models.py --model msgarch
python3 run_user_models.py --model hmm
python3 run_user_models.py --model hsmm
```

---

## 📝 Notes

- **Dependencies**: Ensure all required packages are installed (`pip install -r requirements.txt`)
- **Data**: The runner downloads data automatically via yfinance
- **Outputs**: All model outputs go to `outputs/<model_name>/`
- **Config**: Parameters are passed via temporary YAML files when run from frontend

---

## 🎯 What This Achieves

✅ Your 4 models are fully integrated into the dashboard  
✅ Users can change parameters (ticker, dates, states, features)  
✅ One-click pipeline execution from frontend  
✅ Real-time progress tracking and result visualization  
✅ Ready for PR and merge into main repo  

---

*Last updated: 2026-05-09*
