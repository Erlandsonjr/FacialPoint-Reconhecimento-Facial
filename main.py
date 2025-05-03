from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
import cv2
from pymongo import MongoClient
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)

db = client["test"]
collection = db["foto"]  

@app.post("/reconhecer/")
async def reconhecer(file: UploadFile = File(...), token: str = Header(...)):
    """
    Endpoint para verificar se a codificação do rosto enviado corresponde a codificação do usuário.
    Retorna apenas um booleano em JSON.
    """
    
    dado = collection.find_one({"token": token}, {"_id": 0})
    if not dado:
        raise HTTPException(status_code=401, detail="Token inválido ou usuário não encontrado.")

    codificacao_token = dado["foto"]

    conteudo = await file.read()
    nparr = np.frombuffer(conteudo, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rostos = face_recognition.face_locations(img_rgb)
    if not rostos:
        return False

    codificacoes = face_recognition.face_encodings(img_rgb, rostos)

    for codificacao_rosto in codificacoes:
        correspondencia = face_recognition.compare_faces([codificacao_token], codificacao_rosto, tolerance=0.5)
        if correspondencia[0]:
            return True

    return False
