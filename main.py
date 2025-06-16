from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
from PIL import Image
import io
import traceback
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "FacialPoint API de Reconhecimento Facial"}

@app.post("/reconhecer/")
async def reconhecer(file: UploadFile = File(...), codificacao: list[float] = Form(...)):
    try:
        logger.info("Processando solicitação de reconhecimento")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        image_array = np.array(image)
        
        face_locations = face_recognition.face_locations(image_array)
        if not face_locations:
            logger.warning("Nenhuma face detectada na imagem")
            return {"success": False, "message": "Nenhuma face detectada na imagem"}
        
        face_encoding = face_recognition.face_encodings(image_array, face_locations)[0]
        
        face_to_compare = np.array(codificacao)
        
        face_distances = face_recognition.face_distance([face_to_compare], face_encoding)
        
        confidence = 1.0 - face_distances[0]
        
        if confidence > 0.6:
            return {
                "success": True,
                "match": True,
                "confidence": float(confidence)
            }
        else:
            return {
                "success": True,
                "match": False,
                "confidence": float(confidence)
            }
    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.post("/reconhecer-multiplos/")
async def reconhecer_multiplos(request: Request, file: UploadFile = File(...)):
    try:
        logger.info("Processando solicitação de reconhecimento múltiplo")
        
        form = await request.form()
        codificacoes_por_usuario = {}
        
        for key, value in form.items():
            if key.startswith("codificacao_"):
                parts = key.split("_")
                if len(parts) >= 2:
                    usuario_id = parts[1]
                    
                    if usuario_id not in codificacoes_por_usuario:
                        codificacoes_por_usuario[usuario_id] = []
                    
                    try:
                        valor_float = float(value)
                        codificacoes_por_usuario[usuario_id].append(valor_float)
                    except ValueError:
                        logger.warning(f"Valor não numérico: {value}")
        
        logger.info(f"Processando {len(codificacoes_por_usuario)} usuários")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        image_array = np.array(image)
        face_locations = face_recognition.face_locations(image_array)
        
        if not face_locations:
            logger.warning("Nenhuma face detectada na imagem")
            return {"success": False, "message": "Nenhuma face detectada na imagem"}
        
        face_encoding = face_recognition.face_encodings(image_array, face_locations)[0]
        
        melhor_match = None
        melhor_score = 0.0
        
        for usuario_id, codificacao in codificacoes_por_usuario.items():
            try:
                logger.info(f"Processando codificação para usuário {usuario_id} com {len(codificacao)} elementos")
                
                if len(codificacao) < 10: 
                    logger.warning(f"Poucos elementos para usuário {usuario_id}: {codificacao}")
                    continue
                    
                codificacao_array = np.array(codificacao)
                
                distances = face_recognition.face_distance([codificacao_array], face_encoding)
                
                score = 1.0 - float(distances[0])
                logger.info(f"Usuário {usuario_id}: score {score}")
                
                if score > 0.6 and score > melhor_score:  
                    melhor_match = usuario_id
                    melhor_score = score
            except Exception as e:
                logger.error(f"Erro ao processar usuário {usuario_id}: {str(e)}")
                logger.error(traceback.format_exc())
        
        if melhor_match:
            logger.info(f"Match encontrado: usuário {melhor_match} com score {melhor_score}")
            return {
                "success": True,
                "match": True,
                "usuarioId": melhor_match,
                "confianca": float(melhor_score)
            }
        else:
            logger.info("Nenhum match encontrado")
            return {"success": True, "match": False, "message": "Nenhuma correspondência encontrada"}
    
    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
