from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
import cv2
import os
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

app = FastAPI()

class CustomCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS":
            response = Response()
        else:
            response = await call_next(request)
            
        # Adicionar cabeçalhos CORS em todas as respostas, incluindo erros
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Max-Age"] = "86400"
        
        return response

# Remova o middleware CORS padrão
# app.add_middleware(CORSMiddleware, ...)

# Adicione o middleware personalizado
app.add_middleware(CustomCORSMiddleware)

@app.post("/reconhecer/")
async def reconhecer(file: UploadFile = File(...), codificacao: list = File(...)):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    
    try:
        # Limitar tamanho do arquivo para evitar problemas
        MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB
        conteudo = await file.read(MAX_FILE_SIZE + 1)
        
        if len(conteudo) > MAX_FILE_SIZE:
            return JSONResponse(
                content={"match": False, "error": "Arquivo muito grande (máx. 1MB)"},
                headers=headers
            )

        # Reduzir tamanho da imagem antes do processamento
        nparr = np.frombuffer(conteudo, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(content={"match": False, "error": "Imagem inválida"}, headers=headers)

        # Redimensionar para processar mais rápido
        height, width = img.shape[:2]
        max_dim = 500
        if height > max_dim or width > max_dim:
            scale = max_dim / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Converta a codificação de string para lista se necessário
        if isinstance(codificacao, str):
            import json
            try:
                codificacao = json.loads(codificacao)
            except:
                return JSONResponse(content={"match": False, "error": "Formato de codificação inválido"}, headers=headers)
        
        # Verifique se a codificação é uma lista
        if not isinstance(codificacao, list):
            return JSONResponse(content={"match": False, "error": "Codificação deve ser uma lista"}, headers=headers)

        rostos = face_recognition.face_locations(img_rgb)
        if not rostos:
            return JSONResponse(content={"match": False, "reason": "Nenhum rosto detectado"}, headers=headers)

        codificacoes = face_recognition.face_encodings(img_rgb, rostos)

        for codificacao_rosto in codificacoes:
            correspondencia = face_recognition.compare_faces([codificacao], codificacao_rosto, tolerance=0.5)
            if correspondencia[0]:
                return JSONResponse(content={"match": True}, headers=headers)

        return JSONResponse(content={"match": False, "reason": "Rosto não corresponde à codificação"}, headers=headers)
        
    except Exception as e:
        import traceback
        error_message = str(e)
        error_trace = traceback.format_exc()
        print(f"Erro: {error_message}")
        print(f"Traceback: {error_trace}")
        return JSONResponse(
            status_code=500,
            content={"match": False, "error": error_message},
            headers=headers
        )

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
