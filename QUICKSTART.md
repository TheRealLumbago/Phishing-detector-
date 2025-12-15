# Quick Start Guide - Phishing Email Detector Frontend

## 🚀 Quick Setup (3 Steps)

### Step 1: Install Dependencies
```bash
pip install flask flask-cors
```
Or install all requirements:
```bash
pip install -r requirements.txt
```

### Step 2: Start the Backend Server
```bash
python app.py
```

You should see:
```
============================================================
Loading Phishing Email Detection Models...
============================================================
✓ Loaded LSTM model from lstm_model.keras
✓ Loaded tokenizer from lstm_tokenizer.pkl
✓ All models loaded successfully!

============================================================
Starting Flask server...
============================================================
API will be available at: http://localhost:5000
Frontend should connect to: http://localhost:5000/predict
============================================================
```

### Step 3: Open the Frontend
Simply open `index.html` in your web browser (double-click the file).

**OR** use a local server (recommended):
```bash
# In a new terminal window
python -m http.server 8000
```
Then open: http://localhost:8000/index.html

## 🎯 Using the Frontend

1. **Paste Email Text**: Copy and paste any email content into the text area
2. **Click "Analyze Email"**: The system will analyze the email using the trained LSTM model
3. **View Results**: 
   - See if the email is classified as "Phishing" or "Safe"
   - Check the confidence level and probability scores
   - View the visual progress bar

## 📧 Try Sample Emails

Click the sample email buttons to test:
- **Phishing Example**: Shows a typical phishing email
- **Safe Example**: Shows a legitimate email

## 🔧 Troubleshooting

### Issue: "Error: Failed to analyze email"
- **Solution**: Make sure `app.py` is running (Step 2)
- Check that the server is running on `http://localhost:5000`

### Issue: "Models not loaded"
- **Solution**: Ensure these files exist in the project directory:
  - `lstm_model.keras`
  - `lstm_tokenizer.pkl`
- If missing, run the training cells in `main.ipynb` first

### Issue: Port 5000 already in use
- **Solution**: Change the port in `app.py`:
  ```python
  app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5000 to 5001
  ```
- Then update `API_URL` in `index.html`:
  ```javascript
  const API_URL = 'http://localhost:5001/predict';
  ```

## 📊 Model Information

The frontend uses the **LSTM (Bidirectional LSTM)** model which achieved:
- **Accuracy**: 96.73%
- **ROC-AUC**: 0.9938
- **F1-Score**: 0.9588

This is one of the best-performing models from the comprehensive analysis in `main.ipynb`.

## 🎨 Features

- ✅ Modern, responsive UI design
- ✅ Real-time email analysis
- ✅ Detailed probability scores
- ✅ Confidence level indicators
- ✅ Sample email testing
- ✅ Error handling and user feedback

Enjoy using the Phishing Email Detector! 🛡️

