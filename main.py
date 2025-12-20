from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import google.genai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import io
import json
import traceback
import base64
from io import BytesIO
import os
from pathlib import Path

# Configuração para produção
if os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("HEROKU_APP"):
    # Em produção, serve arquivos estáticos
    from fastapi.staticfiles import StaticFiles
    
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    @app.get("/")
    async def serve_frontend():
        try:
            index_path = Path("static/index.html")
            if index_path.exists():
                with open(index_path, "r", encoding="utf-8") as f:
                    return HTMLResponse(f.read())
        except:
            pass
        return {"message": "SchoolQuest API - Backend Online"}

# =========================================================
# CARREGAMENTO DE VARIÁVEIS
# =========================================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY não encontrada no arquivo .env")

client = genai.Client(api_key=GOOGLE_API_KEY)

# =========================================================
# SISTEMA INTELIGENTE DE MODELOS (NOVA FUNÇÃO)
# =========================================================
def get_available_model():
    """
    Retorna o melhor modelo disponível para o usuário
    """
    try:
        models = client.models.list()
        available_models = [model.name for model in models]
        print(f"📋 Modelos disponíveis: {available_models}")
        
        # ✅ LISTA ATUALIZADA: Modelo comprovadamente funcional em primeiro lugar
        model_priority = [
            "gemini-2.5-flash",           # Modelo que FUNCIONOU no seu teste
            "gemini-2.5-pro",             # Alternativa avançada
            "gemini-2.0-flash",           # Mantém na lista, mas com menor prioridade
            "gemini-2.0-flash-001",
            "gemini-flash-latest",
        ]
        
        for preferred_name in model_priority:
            for available_model in available_models:
                if preferred_name in available_model:
                    print(f"✅ Usando modelo: {available_model}")
                    return available_model
        
        for available_model in available_models:
            if "gemini" in available_model.lower():
                print(f"⚠️ Usando fallback: {available_model}")
                return available_model
                
    except Exception as e:
        print(f"⚠️ Não foi possível listar modelos: {e}")
    
    print("⚠️ Usando fallback padrão: models/gemini-2.5-flash")
    return "models/gemini-2.5-flash"

# Obter o modelo correto
MODEL_NAME = get_available_model()

# =========================================================
# CONFIGURAÇÃO DO MODELO
# =========================================================
generation_config = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 4096,
}

# =========================================================
# APP FASTAPI
# =========================================================
app = FastAPI(
    title="SchoolQuest API",
    version="1.0.0",
    description="API gratuita de gamificação escolar para crianças"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# MODELOS
# =========================================================
class TextInput(BaseModel):
    text: str

# =========================================================
# FUNÇÕES UTILITÁRIAS
# =========================================================
def safe_json_parse(text: str):
    """
    Garante que a resposta da IA seja um JSON válido
    """
    try:
        cleaned = text.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "")
        cleaned = cleaned.replace("json\n", "")
        
        # Tenta parsear normalmente
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao parsear JSON: {e}")
        print(f"📝 Tamanho do texto: {len(text)} caracteres")
        print(f"📝 Últimos 200 caracteres: {text[-200:]}")
        
        # Tenta encontrar JSON incompleto e completar
        if '"questions": [' in text and not text.strip().endswith("}]}"):
            print("⚠️  JSON parece incompleto. Tentando completar...")
            
            # Adiciona colchetes/chaves faltantes
            open_braces = text.count('{')
            close_braces = text.count('}')
            open_brackets = text.count('[')
            close_brackets = text.count(']')
            
            if open_braces > close_braces:
                text += '}' * (open_braces - close_braces)
            if open_brackets > close_brackets:
                text += ']' * (open_brackets - close_brackets)
            
            # Garante que termina corretamente
            if not text.strip().endswith('}'):
                text = text.rstrip() + '}'
            
            print(f"🔧 Texto corrigido (últimos 100): {text[-100:]}")
            
            try:
                return json.loads(text)
            except json.JSONDecodeError as e2:
                print(f"❌ Ainda não é JSON válido após correção: {e2}")
        
        raise ValueError("Resposta da IA não é um JSON válido")

def validate_questions(game_data: dict):
    if "questions" not in game_data or not game_data["questions"]:
        raise ValueError("Nenhuma questão foi gerada")

    if len(game_data["questions"]) > 10:
        game_data["questions"] = game_data["questions"][:10]

    for i, q in enumerate(game_data["questions"]):
        required = ["question", "options", "correct", "explanation", "points"]
        if not all(k in q for k in required):
            raise ValueError(f"Questão {i+1} com campos faltando")

        if len(q["options"]) != 4:
            raise ValueError(f"Questão {i+1} deve ter 4 opções")

        if not (0 <= q["correct"] <= 3):
            raise ValueError(f"Questão {i+1} com índice inválido")

# =========================================================
# ROTAS
# =========================================================
@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>SchoolQuest API</h1><p>Backend ativo 🚀</p>"

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "gemini_configured": True,
        "model": MODEL_NAME,
        "api_key_set": bool(GOOGLE_API_KEY)
    }

# =========================================================
# PROCESSAMENTO DE IMAGEM
# =========================================================
@app.post("/api/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        print(f"🖼️ Processando imagem com modelo: {MODEL_NAME}")
        
        contents = await file.read()

        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(400, "Imagem muito grande (máx 5MB)")

        image = Image.open(io.BytesIO(contents))

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Converter imagem para base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        prompt = """Analise esta imagem de dever de casa escolar e crie 5 a 10 questões DIVERTIDAS para uma criança de 8-9 anos.

IMPORTANTE: Responda APENAS com JSON válido, sem texto extra, sem markdown, sem código.

Formato EXATO do JSON:
{
  "questions": [
    {
      "question": "Pergunta com emoji 😊",
      "options": ["Opção A", "Opção B", "Opção C", "Opção D"],
      "correct": 0,
      "explanation": "Explicação educativa e motivadora",
      "points": 15,
      "difficulty": "fácil"
    }
  ]
}

Use linguagem simples, emojis e criatividade."""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_base64
                            }
                        }
                    ]
                }
            ],
            config=generation_config
        )
        
        print(f"✅ Resposta recebida ({len(response.text)} caracteres): {response.text[:150]}...")
        
        # VERIFICAÇÃO EXTRA: Salva resposta para depuração
        with open("debug_response.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
        print("📁 Resposta salva em debug_response.txt para análise")
        
        game_data = safe_json_parse(response.text)
        validate_questions(game_data)

        return JSONResponse(content=game_data)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar imagem: {str(e)}"
        )

# =========================================================
# PROCESSAMENTO DE TEXTO
# =========================================================
@app.post("/api/process-text")
async def process_text(data: TextInput):
    try:
        print(f"📝 Processando texto com modelo: {MODEL_NAME}")
        
        prompt = f"""Baseado neste texto de dever de casa, crie 5 a 10 questões gamificadas para uma criança de 8-9 anos:

{data.text}

IMPORTANTE: Responda APENAS com JSON válido, sem texto extra, sem markdown, sem código.

Formato EXATO do JSON:
{{
  "questions": [
    {{
      "question": "Pergunta com emoji 😊",
      "options": ["Opção A", "Opção B", "Opção C", "Opção D"],
      "correct": 0,
      "explanation": "Explicação motivadora",
      "points": 15,
      "difficulty": "fácil"
    }}
  ]
}}"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config=generation_config
        )
        
        print(f"✅ Resposta recebida: {response.text[:100]}...")
        game_data = safe_json_parse(response.text)
        validate_questions(game_data)

        return JSONResponse(content=game_data)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar texto: {str(e)}"
        )

# =========================================================
# START SERVER
# =========================================================
if __name__ == "__main__":
    import uvicorn

    print("🎮 SchoolQuest API iniciando...")
    print("📱 Mobile-first | 🤖 Gemini | 🎓 Gamificação")
    print(f"🤖 Modelo selecionado: {MODEL_NAME}")
    print("📡 http://localhost:8000")
    print("📘 Docs: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)