from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import traceback
import io
from PIL import Image
import numpy as np
import face_recognition

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Para desenvolvimento. Em produção, especifique origens exatas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/reconhecer/")
async def reconhecer(file: UploadFile = File(...), codificacao: List[float] = Body(...)):
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

@app.post("/reconhecer-multiplos/")
async def reconhecer_multiplos(file: UploadFile = File(...)):
    try:
        # Extrair dados da requisição
        form = await request.form()
        codificacoes_por_usuario = {}
        
        # Processar todas as codificações enviadas
        for key, value in form.items():
            if key.startswith("codificacao_"):
                # Formato da chave: codificacao_USERID_INDEX
                parts = key.split("_")
                if len(parts) == 3:
                    usuario_id = parts[1]
                    
                    if usuario_id not in codificacoes_por_usuario:
                        codificacoes_por_usuario[usuario_id] = []
                    
                    try:
                        valor_float = float(value)
                        codificacoes_por_usuario[usuario_id].append(valor_float)
                    except ValueError:
                        print(f"Valor não numérico: {value}")
        
        # Processar a imagem enviada
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Detectar face na imagem
        face_locations = face_recognition.face_locations(np.array(image))
        if not face_locations:
            return {"success": False, "message": "Nenhuma face detectada na imagem"}
        
        # Codificar a face detectada
        face_encoding = face_recognition.face_encodings(np.array(image), face_locations)[0]
        
        # Comparar com todas as codificações de todos os usuários
        melhor_match = None
        melhor_score = 0.0
        
        for usuario_id, codificacao in codificacoes_por_usuario.items():
            # Verificar se a codificação tem tamanho correto
            if len(codificacao) != 128:
                print(f"Codificação para usuário {usuario_id} tem tamanho incorreto: {len(codificacao)}")
                continue
            
            # Reshape para formato esperado pelo face_recognition
            codificacao_array = np.array(codificacao)
            
            # Calcular a distância (menor é melhor)
            distances = face_recognition.face_distance([codificacao_array], face_encoding)
            
            # Converter distância para score (maior é melhor)
            score = 1.0 - distances[0]
            
            if score > 0.6 and score > melhor_score:  # 0.6 é um limite aceitável
                melhor_match = usuario_id
                melhor_score = score
        
        if melhor_match:
            return {
                "success": True,
                "match": True,
                "usuarioId": melhor_match,
                "confianca": float(melhor_score)
            }
        else:
            return {"success": True, "match": False, "message": "Nenhuma correspondência encontrada"}
    
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Erro: {str(e)}")
        print(traceback_str)
        return {"success": False, "error": str(e), "traceback": traceback_str}

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

@app.get("/health")
async def health_check():
    """
    Endpoint simples para verificação de saúde do serviço.
    """
    return {"status": "healthy"}
