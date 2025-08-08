import os
from fastapi import FastAPI, Request, HTTPException
from formula_runner import optimize_sheet

app = FastAPI()
API_KEY = os.environ.get("API_KEY")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/run")
async def run(request: Request):
    if API_KEY and request.headers.get("X-API-KEY") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    sheet_id = body.get("sheetId")
    input_ws = body.get("inputSheet", "Input")
    sku_ws = body.get("skuSheet", "SKU")
    out_ws = body.get("outputSheet", "Optimized Formula")

    if not sheet_id:
        raise HTTPException(status_code=400, detail="Missing sheetId")

    result = optimize_sheet(sheet_id, input_ws, sku_ws, out_ws)
    return result
