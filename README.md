## 🗂️ Project Structure

```bash
network-attack-system/
├── network-attack-backend/        # FastAPI backend
│   ├── main.py                    # Main API server
│   ├── train_model.py             # Model training script
│   ├── requirements.txt           # Python dependencies
│   └── model_files/               # Trained ML models
│
├── network-attack-frontend/       # React frontend
│   ├── src/
│   ├── public/
│   └── package.json               # Node dependencies
│
├── run.ps1                        # One-click startup script
└── README.md


⚡ Quick Start (Recommended)
✅ Prerequisites

Python 3.8+

Node.js 14+

Git

▶️ Run Everything with One Command (Windows)
cd network-attack-system
.\run.ps1


This will automatically:

Create a Python virtual environment

Install backend dependencies

Install frontend dependencies

Start backend at http://localhost:8000

Start frontend at http://localhost:3000

🔧 Manual Setup
🖥️ Backend (FastAPI)
cd network-attack-backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --reload


Backend runs at:

http://localhost:8000

🌐 Frontend (React)
cd network-attack-frontend
npm install
npm start


Frontend runs at:

http://localhost:3000

🔐 Environment Configuration

Create a .env file inside network-attack-backend/

GEMINI_API_KEY=your_api_key_here

🔌 API Endpoints
Method	Endpoint	Description
POST	/predict	Predict attack from CSV upload
POST	/predict-manual	Predict using manual input
POST	/batch-predict	Batch predictions
POST	/mitigation	Get AI-based mitigation suggestions
📦 Requirements
Backend

FastAPI

scikit-learn

pandas

numpy

uvicorn

Frontend

React

Axios

Chart.js

(See requirements.txt and package.json for full list)

🛑 Stop Services

Press Ctrl + C in each terminal window.

📄 License

MIT License
