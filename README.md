# FacialPoint — Serviço de Reconhecimento Facial

Microserviço responsável por comparar o rosto capturado em tempo real com as codificações faciais armazenadas dos usuários cadastrados.

## Sobre o projeto

O FacialPoint é um sistema de registro de ponto por reconhecimento facial. Este serviço recebe uma imagem e uma lista de codificações faciais, identifica o rosto presente na imagem e retorna qual usuário foi reconhecido com a respectiva pontuação de confiança.

## Tecnologias

- **Python 3.10**
- **FastAPI** — framework web assíncrono
- **face_recognition** — reconhecimento facial baseado em dlib
- **NumPy** — operações com vetores de codificação
- **Pillow** — processamento de imagens
- **Docker** — containerização

## Funcionalidades

- Reconhecimento facial contra um único usuário (`/reconhecer/`)
- Reconhecimento facial contra múltiplos usuários simultaneamente (`/reconhecer-multiplos/`)
- Retorno da confiança do reconhecimento (0 a 1)
- Threshold configurável de similaridade (padrão: 0.5)

## Como executar

**Localmente**
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Com Docker**
```bash
docker build -t facialpoint-reconhecimento .
docker run -p 8000:8000 facialpoint-reconhecimento
```

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `POST` | `/reconhecer/` | Compara rosto com uma codificação |
| `POST` | `/reconhecer-multiplos/` | Compara rosto com múltiplas codificações |

## Repositórios relacionados

- [FacialPoint-Site](https://github.com/Erlandsonjr/FacialPoint-Site) — Interface web (React)
- [FacialPoint-Banco-Dados](https://github.com/Erlandsonjr/FacialPoint-Banco-Dados) — API backend (Node.js)
- [gerar_codificacao_facialpoint](https://github.com/Erlandsonjr/gerar_codificacao_facialpoint) — Serviço de codificação facial (Python)
