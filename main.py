from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from PIL import Image
import io
import json
import traceback
import base64
from io import BytesIO
from pathlib import Path
import hashlib
import time
from openai import OpenAI

# =========================================================
# CARREGAMENTO DE VARIÁVEIS
# =========================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY não encontrada no arquivo .env")

print("\n🤖 Provedor de IA: OPENAI")

# =========================================================
# INICIALIZAÇÃO DO CLIENTE OPENAI
# =========================================================
client = OpenAI(api_key=OPENAI_API_KEY)
print("✅ Cliente OpenAI inicializado")

# Modelo mais poderoso da OpenAI
MODEL_NAME = "gpt-4o"
print(f"✅ Usando modelo: {MODEL_NAME} 🏆")

# =========================================================
# CACHE SIMPLES EM MEMÓRIA
# =========================================================
class SimpleCache:
    def __init__(self, ttl=86400):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                print(f"✅ Cache HIT: {key[:20]}...")
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = (value, time.time())
        print(f"💾 Cache SAVE: {key[:20]}...")
    
    def clear_old(self):
        now = time.time()
        expired = [k for k, (_, ts) in self.cache.items() if now - ts > self.ttl]
        for k in expired:
            del self.cache[k]
        if expired:
            print(f"🧹 Cache limpo: {len(expired)} entradas removidas")

api_cache = SimpleCache(ttl=86400)

# =========================================================
# CONFIGURAÇÃO DO MODELO
# =========================================================
generation_config = {
    "temperature": 0.7,
    "max_tokens": 8192,
}

# =========================================================
# APP FASTAPI
# =========================================================
app = FastAPI(
    title="SchoolQuest API",
    version="3.0.0",
    description="API gamificada com OpenAI"
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

class ShuffleInput(BaseModel):
    questions: list

# =========================================================
# FUNÇÕES UTILITÁRIAS
# =========================================================
def generate_cache_key(content: str, content_type: str = "text") -> str:
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return f"openai_{content_type}_{content_hash}"

def safe_json_parse(text: str):
    try:
        cleaned = text.strip()
        
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if len(lines) > 2 else lines)
        
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao parsear JSON: {e}")
        print(f"📝 Texto recebido (primeiros 500 chars):\n{text[:500]}")
        
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_text = text[start_idx:end_idx + 1]
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Resposta da IA não é um JSON válido. Erro: {str(e)}")

def validate_questions(game_data: dict):
    if "questions" not in game_data or not game_data["questions"]:
        raise ValueError("Nenhuma questão foi gerada")

    if len(game_data["questions"]) > 10:
        game_data["questions"] = game_data["questions"][:10]

    for i, q in enumerate(game_data["questions"]):
        required = ["question", "options", "correct", "explanation"]
        missing = [field for field in required if field not in q]
        
        if missing:
            raise ValueError(f"Questão {i+1} está faltando: {', '.join(missing)}")

        if not isinstance(q["options"], list) or len(q["options"]) != 4:
            raise ValueError(f"Questão {i+1} deve ter exatamente 4 opções em lista")

        if not isinstance(q["correct"], int) or not (0 <= q["correct"] <= 3):
            raise ValueError(f"Questão {i+1} tem índice 'correct' inválido: {q.get('correct')}")
        
        if "difficulty" not in q:
            q["difficulty"] = "médio"
        
        if "points" not in q:
            difficulty_points = {"fácil": 10, "médio": 15, "difícil": 20}
            q["points"] = difficulty_points.get(q["difficulty"], 15)

def create_game_prompt(content_description: str = "") -> str:
    if content_description:
        prompt = f"""Você é um assistente educacional especializado em criar questões de múltipla escolha divertidas e educativas para crianças de 8-9 anos.

**SUA TAREFA**: Analise o conteúdo abaixo e crie questões ESPECIFICAMENTE sobre os tópicos, conceitos e informações presentes nesse conteúdo.

**CONTEÚDO DO DEVER DE CASA**:
{content_description}

**IMPORTANTE**: 
- Crie questões APENAS sobre o conteúdo acima
- Se for matemática, faça questões de matemática
- Se for português, faça questões de português
- Se for ciências, faça questões de ciências
- Se for história/geografia, faça questões dessas matérias
- Use os números, conceitos e informações EXATOS do conteúdo

**FORMATO DE RESPOSTA** - Responda APENAS com um objeto JSON válido:

{{
  "questions": [
    {{
      "question": "Pergunta sobre o conteúdo com emoji 😊",
      "options": ["Opção A", "Opção B", "Opção C", "Opção D"],
      "correct": 0,
      "explanation": "Explicação educativa",
      "points": 15,
      "difficulty": "médio"
    }}
  ]
}}

**REGRAS**:
1. Use linguagem SIMPLES para crianças de 8-9 anos
2. Inclua emojis nas perguntas
3. Crie 5 a 10 questões SOBRE O CONTEÚDO ENVIADO
4. Cada questão: exatamente 4 opções
5. Campo "correct": número de 0 a 3
6. Dificuldade: fácil (10 pontos), médio (15 pontos), difícil (20 pontos)

**AGORA GERE O JSON** (sem texto adicional):"""
    else:
        prompt = """Você é um assistente educacional. Crie 5 questões educativas variadas para crianças de 8-9 anos.

Responda APENAS com JSON:

{
  "questions": [
    {
      "question": "Pergunta com emoji 😊",
      "options": ["Opção A", "Opção B", "Opção C", "Opção D"],
      "correct": 0,
      "explanation": "Explicação",
      "points": 15,
      "difficulty": "médio"
    }
  ]
}"""
    
    return prompt

# =========================================================
# FUNÇÕES DE CHAMADA À IA (OPENAI)
# =========================================================
def call_ai_with_text(prompt: str) -> str:
    """Chama a OpenAI com texto"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=generation_config["temperature"],
        max_tokens=generation_config["max_tokens"]
    )
    return response.choices[0].message.content

def call_ai_with_image(prompt: str, image_base64: str) -> str:
    """Chama a OpenAI com imagem"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        temperature=generation_config["temperature"],
        max_tokens=generation_config["max_tokens"]
    )
    return response.choices[0].message.content

# =========================================================
# ROTAS
# =========================================================
@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        index_path = Path("index.html")
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        
        index_path = Path("static/index.html")
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
    except Exception as e:
        print(f"⚠️ Erro ao carregar index.html: {e}")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SchoolQuest API</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            .container {{
                background: white;
                color: #2D3748;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }}
            h1 {{ color: #667eea; margin-bottom: 10px; }}
            .status {{ color: #06D6A0; font-weight: bold; font-size: 20px; }}
            .provider {{ 
                background: linear-gradient(135deg, #10a37f, #1a7f64);
                color: white;
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
            }}
            .endpoint {{ 
                background: #F7FAFC; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 10px;
                border-left: 4px solid #10a37f;
            }}
            a {{ color: #10a37f; text-decoration: none; font-weight: bold; }}
            a:hover {{ text-decoration: underline; }}
            code {{ 
                background: #2D3748; 
                color: #06D6A0; 
                padding: 2px 8px; 
                border-radius: 4px;
                font-family: monospace;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎮 SchoolQuest API v3.0.0</h1>
            <p class="status">✅ Backend Online e Funcionando!</p>
            
            <div class="provider">
                🤖 OpenAI - {MODEL_NAME}
            </div>
            
            <h2>📚 Endpoints Disponíveis:</h2>
            
            <div class="endpoint">
                <strong>📘 Documentação Interativa:</strong><br>
                <a href="/docs" target="_blank">/docs</a>
            </div>
            
            <div class="endpoint">
                <strong>🏥 Health Check:</strong><br>
                <a href="/api/health" target="_blank">/api/health</a>
            </div>
            
            <div class="endpoint">
                <strong>🖼️ Processar Imagem:</strong><br>
                <code>POST /api/process-image</code>
            </div>
            
            <div class="endpoint">
                <strong>📝 Processar Texto:</strong><br>
                <code>POST /api/process-text</code>
            </div>
            
            <div class="endpoint">
                <strong>📊 Estatísticas do Cache:</strong><br>
                <a href="/api/cache/stats" target="_blank">/api/cache/stats</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/health")
async def health():
    api_cache.clear_old()
    
    return {
        "status": "healthy",
        "ai_provider": "openai",
        "model": MODEL_NAME,
        "api_key_set": bool(OPENAI_API_KEY),
        "cache_entries": len(api_cache.cache),
        "cache_ttl_hours": api_cache.ttl / 3600,
        "version": "3.0.0",
        "timestamp": time.time(),
        "features": {
            "text_processing": True,
            "image_processing": True
        }
    }

@app.get("/api/cache/clear")
async def clear_cache():
    entries = len(api_cache.cache)
    api_cache.cache.clear()
    return {
        "status": "ok",
        "message": f"Cache limpo com sucesso!",
        "entries_removed": entries
    }

@app.get("/api/cache/stats")
async def cache_stats():
    api_cache.clear_old()
    
    total_entries = len(api_cache.cache)
    image_entries = sum(1 for k in api_cache.cache.keys() if "image" in k)
    text_entries = sum(1 for k in api_cache.cache.keys() if "text" in k)
    
    return {
        "ai_provider": "openai",
        "model": MODEL_NAME,
        "total_entries": total_entries,
        "image_entries": image_entries,
        "text_entries": text_entries,
        "ttl_hours": api_cache.ttl / 3600
    }

@app.post("/api/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        print(f"\n{'='*60}")
        print(f"🖼️ Processando imagem: {file.filename}")
        print(f"🤖 Modelo: {MODEL_NAME}")
        print(f"{'='*60}\n")
        
        contents = await file.read()

        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(400, "Imagem muito grande. Máximo: 5MB")

        img_hash = hashlib.md5(contents).hexdigest()
        cache_key = generate_cache_key(img_hash, "image")
        
        cached_result = api_cache.get(cache_key)
        if cached_result:
            print("✅ Resultado recuperado do cache (tokens economizados!)")
            return JSONResponse(content=cached_result)

        print("📸 Processando imagem (primeira vez)...")

        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")

        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85, optimize=True)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        prompt = create_game_prompt("Analise esta imagem de um dever de casa e crie questões sobre o conteúdo presente na imagem.")

        print("🚀 Enviando para OpenAI...")

        response_text = call_ai_with_image(prompt, img_base64)
        
        print(f"✅ Resposta recebida ({len(response_text)} caracteres)")
        
        game_data = safe_json_parse(response_text)
        validate_questions(game_data)
        
        print(f"✅ {len(game_data['questions'])} questões geradas com sucesso!")
        
        api_cache.set(cache_key, game_data)

        return JSONResponse(content=game_data)

    except HTTPException:
        raise
    except ValueError as ve:
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar imagem: {str(e)}"
        )

@app.post("/api/process-text")
async def process_text(data: TextInput):
    try:
        print(f"\n{'='*60}")
        print(f"📝 Processando texto ({len(data.text)} caracteres)")
        print(f"🤖 Modelo: {MODEL_NAME}")
        print(f"{'='*60}\n")
        
        if not data.text or len(data.text.strip()) < 10:
            raise HTTPException(400, "Texto muito curto. Mínimo: 10 caracteres")
        
        cache_key = generate_cache_key(data.text, "text")
        
        cached_result = api_cache.get(cache_key)
        if cached_result:
            print("✅ Resultado recuperado do cache (tokens economizados!)")
            return JSONResponse(content=cached_result)

        print("📄 Processando texto (primeira vez)...")

        prompt = create_game_prompt(f"**Conteúdo do dever de casa**:\n\n{data.text}")

        print("🚀 Enviando para OpenAI...")

        response_text = call_ai_with_text(prompt)
        
        print(f"✅ Resposta recebida ({len(response_text)} caracteres)")
        
        game_data = safe_json_parse(response_text)
        validate_questions(game_data)
        
        print(f"✅ {len(game_data['questions'])} questões geradas com sucesso!")
        
        api_cache.set(cache_key, game_data)

        return JSONResponse(content=game_data)

    except HTTPException:
        raise
    except ValueError as ve:
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar texto: {str(e)}"
        )

@app.post("/api/shuffle-questions")
async def shuffle_questions(data: ShuffleInput):
    try:
        import random
        
        if not data.questions or len(data.questions) == 0:
            raise HTTPException(400, "Nenhuma questão fornecida para embaralhar")
        
        shuffled = data.questions.copy()
        random.shuffle(shuffled)
        
        for q in shuffled:
            if "options" in q and "correct" in q:
                correct_answer = q["options"][q["correct"]]
                random.shuffle(q["options"])
                q["correct"] = q["options"].index(correct_answer)
        
        print(f"🔀 Embaralhadas {len(shuffled)} questões")
        
        return JSONResponse(content={"questions": shuffled})
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Erro ao embaralhar: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    
    print("\n" + "="*60)
    print("🎮 SchoolQuest API v3.0.0")
    print("="*60)
    print(f"🤖 Provedor de IA: OPENAI")
    print(f"📦 Modelo ativo: {MODEL_NAME}")
    print("💾 Cache: Ativado (24 horas)")
    print("🔒 CORS: Habilitado")
    print("="*60)
    print(f"📡 Servidor: http://0.0.0.0:{port}")
    print("📘 Documentação: /docs")
    print("🏥 Health check: /api/health")
    print("="*60 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )