from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
import cv2
import os
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

@app.post("/reconhecer/")
async def reconhecer(file: UploadFile = File(...), codificacao: list = File(...)):
    """
    Endpoint para verificar se a codificação do rosto enviado corresponde à codificação fornecida.
    Retorna apenas um booleano em JSON.
    """

    conteudo = await file.read()
    nparr = np.frombuffer(conteudo, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rostos = face_recognition.face_locations(img_rgb)
    if not rostos:
        return JSONResponse(content={"match": False}, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        })

    codificacoes = face_recognition.face_encodings(img_rgb, rostos)

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    for codificacao_rosto in codificacoes:
        correspondencia = face_recognition.compare_faces([codificacao], codificacao_rosto, tolerance=0.5)
        if correspondencia[0]:
            return JSONResponse(content={"match": True}, headers=headers)

    return JSONResponse(content={"match": False}, headers=headers)

@app.options("/reconhecer/")
async def options_reconhecer():
    # Responder a requisições OPTIONS com cabeçalhos CORS apropriados
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content={"msg": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400",  # 24 horas em segundos
        },
    )
