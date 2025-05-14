from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load the model
model = joblib.load("svc_model_11_features.pkl")

app = FastAPI()

# CORS setup for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your mobile app domain if deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input format
class LoanInput(BaseModel):
    Gender: int
    Married: int
    Dependents: int
    Education: int
    Self_Employed: int
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: int
    Property_Area: int

@app.post("/predict")
def predict(data: LoanInput):
    input_data = np.array([[data.Gender, data.Married, data.Dependents, data.Education,
                            data.Self_Employed, data.ApplicantIncome, data.CoapplicantIncome,
                            data.LoanAmount, data.Loan_Amount_Term, data.Credit_History,
                            data.Property_Area]])
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
