from fastapi import FastAPI, Request
from datetrans.convert_extract import converter

app = FastAPI()

@app.post("/")
async def main(message_json:str):
    response = converter(message_json)
    return {"response":response}
