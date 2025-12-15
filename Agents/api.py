from fastapi import FastAPI, HTTPException
from market_agent import MarketActivityAgent

app = FastAPI(title="Market Activity Prediction Agent")
agent = MarketActivityAgent()

@app.get("/health")
def health_check():
    return {"status": "active"}

@app.get("/predict/{ticker}")
def predict_market_activity(ticker: str):
    try:
        result = agent.analyze(ticker.upper())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn api:app --reload