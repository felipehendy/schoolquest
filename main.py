from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
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
import jwt
from datetime import datetime, timedelta
from typing import Optional

# =========================================================
# CARREGAMENTO DE VARIÁVEIS
# =========================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "sua-chave-secreta-super-segura-mude-isso")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY não encontrada no arquivo .env")

print("\n🤖 Provedor de IA: OPENAI")

# =========================================================
# INICIALIZAÇÃO DO CLIENTE OPENAI
# =========================================================
client = OpenAI(api_key=OPENAI_API_KEY)
print("✅ Cliente OpenAI inicializado")

MODEL_NAME = "gpt-4o"
print(f"✅ Usando modelo: {MODEL_NAME} 🏆")

# =========================================================
# BANCO DE DADOS SIMPLES (EM MEMÓRIA)
# =========================================================
users_db = {}

# =========================================================
# SEGURANÇA
# =========================================================
security = HTTPBearer(auto_error=False)

def create_token(username: str) -> str:
    payload = {
        "username": username,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Token não fornecido")
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["username"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expirado")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token inválido")

def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Permite acesso com ou sem autenticação (modo visitante)"""
    if credentials is None:
        return "guest"
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload.get("username", "guest")
    except jwt.PyJWTError:
        return "guest"

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
    version="4.0.0",
    description="API gamificada com OpenAI e autenticação"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# SERVIR ARQUIVOS ESTÁTICOS
# =========================================================
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================================================
# MODELOS
# =========================================================
class LoginInput(BaseModel):
    username: str
    password: str

class RegisterInput(BaseModel):
    username: str
    password: str
    email: str = None

class TextInput(BaseModel):
    text: str

class ShuffleInput(BaseModel):
    questions: list

# =========================================================
# FUNÇÕES UTILITÁRIAS
# =========================================================
def get_html_file(filename: str) -> HTMLResponse:
    """Busca arquivo HTML na raiz ou na pasta static (compatível com Render)"""
    # Tenta primeiro na raiz (desenvolvimento local)
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    
    # Tenta na pasta static (Render)
    static_path = os.path.join("static", filename)
    if os.path.exists(static_path):
        with open(static_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    
    raise HTTPException(status_code=404, detail=f"Arquivo {filename} não encontrado")

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

    if len(game_data["questions"]) > 20:
        game_data["questions"] = game_data["questions"][:20]

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
        prompt = f"""Você é um PROFESSOR PEDAGOGO ESPECIALISTA em ensino infantil (8 a 10 anos),
com foco em aprendizagem ativa, raciocínio lógico, criatividade e gamificação.

Você trabalha para uma plataforma educacional chamada SCHOOLQUEST,
onde o aprendizado acontece por meio de DESAFIOS e JOGOS.

══════════════════════════════════════
🎯 ETAPA 1 — IDENTIFICAÇÃO DA MATÉRIA
══════════════════════════════════════
Analise o conteúdo abaixo e IDENTIFIQUE AUTOMATICAMENTE a matéria principal.

Matérias possíveis:
- Matemática
- Português
- Ciências
- História
- Geografia
- Inglês
- Conhecimentos Gerais

══════════════════════════════════════
🎯 ETAPA 2 — REGRAS POR MATÉRIA
══════════════════════════════════════

📘 SE A MATÉRIA FOR **MATEMÁTICA**:
- NÃO faça perguntas de interpretação de texto
- NÃO conte letras ou palavras
- CRIE CÁLCULOS NOVOS, mesmo que o texto não tenha números
- Use obrigatoriamente:
  • soma, subtração, multiplicação, divisão simples
- Crie situações do cotidiano infantil:
  • dinheiro, brinquedos, frutas, tempo, escola
- Exija raciocínio lógico e cálculo mental

📗 SE A MATÉRIA FOR **PORTUGUÊS**:
- Trabalhe: interpretação de texto, ortografia, sinônimos e antônimos, gramática básica
- Pode criar exemplos novos além do texto

📙 SE A MATÉRIA FOR **CIÊNCIAS**:
- Use perguntas sobre: corpo humano, natureza, animais, meio ambiente
- Linguagem simples e educativa

📕 SE A MATÉRIA FOR **HISTÓRIA**:
- Perguntas sobre: fatos históricos, personagens, datas importantes
- Sempre contextualizadas

📒 SE A MATÉRIA FOR **GEOGRAFIA**:
- Trabalhe: mapas, países, estados, clima, natureza
- Use exemplos do cotidiano

📔 SE A MATÉRIA FOR **INGLÊS**:
- Use palavras simples
- Trabalhe: cores, números, animais, objetos
- Pode misturar português + inglês

══════════════════════════════════════
📚 CONTEÚDO DO ALUNO
══════════════════════════════════════
{content_description}

══════════════════════════════════════
🎯 FORMATO DE SAÍDA (JSON PURO)
══════════════════════════════════════

Retorne SOMENTE JSON válido, sem texto explicativo antes ou depois.

**FORMATO OBRIGATÓRIO**:

{{
  "questions": [
    {{
      "question": "Pergunta com emoji 😊",
      "options": ["Opção A", "Opção B", "Opção C", "Opção D"],
      "correct": 0,
      "explanation": "Explicação educativa clara",
      "points": 15,
      "difficulty": "médio"
    }}
  ]
}}

**REGRAS FINAIS**:
1. Use linguagem SIMPLES para crianças de 8-10 anos
2. Inclua emojis nas perguntas para deixar divertido
3. Crie entre 10 a 20 questões SOBRE O CONTEÚDO ENVIADO
4. Cada questão: exatamente 4 opções
5. Campo "correct": número de 0 a 3 (índice da resposta correta)
6. Dificuldade: "fácil" (10 pontos), "médio" (15 pontos), "difícil" (20 pontos)
7. Explicação: clara, educativa e encorajadora

**AGORA GERE O JSON**:"""
    else:
        prompt = """Você é um assistente educacional. Crie 10 questões educativas variadas para crianças de 8-10 anos.

Responda APENAS com JSON válido:

{
  "questions": [
    {
      "question": "Pergunta com emoji 😊",
      "options": ["Opção A", "Opção B", "Opção C", "Opção D"],
      "correct": 0,
      "explanation": "Explicação clara e educativa",
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
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=generation_config["temperature"],
        max_tokens=generation_config["max_tokens"]
    )
    return response.choices[0].message.content

def call_ai_with_image(prompt: str, image_base64: str) -> str:
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
# ROTAS PRINCIPAIS
# =========================================================
@app.get("/")
async def root():
    """Redireciona para a página de login"""
    return RedirectResponse(url="/login.html")

@app.get("/login.html")
async def serve_login():
    """Serve a página de login"""
    return get_html_file("login.html")

@app.get("/register.html")
async def serve_register():
    """Serve a página de registro"""
    return get_html_file("register.html")

@app.get("/index.html")
async def serve_index():
    """Serve a página principal do jogo"""
    return get_html_file("index.html")

# =========================================================
# ROTAS DE AUTENTICAÇÃO
# =========================================================
@app.post("/api/auth/register")
async def register(data: RegisterInput):
    if len(data.username) < 3:
        raise HTTPException(400, "Usuário deve ter pelo menos 3 caracteres")
    
    if len(data.password) < 4:
        raise HTTPException(400, "Senha deve ter pelo menos 4 caracteres")
    
    if data.username in users_db:
        raise HTTPException(400, "Usuário já existe")
    
    password_hash = hashlib.sha256(data.password.encode()).hexdigest()
    
    users_db[data.username] = {
        "password": password_hash,
        "email": data.email,
        "created_at": datetime.utcnow().isoformat()
    }
    
    print(f"✅ Novo usuário registrado: {data.username}")
    
    return {
        "message": "Usuário criado com sucesso",
        "username": data.username
    }

@app.post("/api/auth/login")
async def login(data: LoginInput):
    if data.username not in users_db:
        raise HTTPException(401, "Usuário ou senha incorretos")
    
    password_hash = hashlib.sha256(data.password.encode()).hexdigest()
    
    if users_db[data.username]["password"] != password_hash:
        raise HTTPException(401, "Usuário ou senha incorretos")
    
    token = create_token(data.username)
    
    print(f"✅ Login realizado: {data.username}")
    
    return {
        "token": token,
        "username": data.username
    }

@app.get("/api/auth/me")
async def get_current_user(username: str = Depends(verify_token)):
    if username not in users_db:
        raise HTTPException(404, "Usuário não encontrado")
    
    return {
        "username": username,
        "email": users_db[username].get("email"),
        "created_at": users_db[username].get("created_at")
    }

# =========================================================
# ROTAS DE MONITORAMENTO
# =========================================================
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
        "version": "4.0.0",
        "timestamp": time.time(),
        "users_count": len(users_db),
        "features": {
            "text_processing": True,
            "image_processing": True,
            "authentication": True,
            "guest_mode": True
        }
    }

@app.get("/api/cache/clear")
async def clear_cache(username: str = Depends(verify_token)):
    entries = len(api_cache.cache)
    api_cache.cache.clear()
    return {
        "status": "ok",
        "message": "Cache limpo com sucesso!",
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

# =========================================================
# ROTAS DE PROCESSAMENTO
# =========================================================
@app.post("/api/process-image")
async def process_image(file: UploadFile = File(...), username: str = Depends(get_optional_user)):
    try:
        print(f"\n{'='*60}")
        print(f"🖼️ Processando imagem: {file.filename}")
        print(f"👤 Usuário: {username}")
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
async def process_text(data: TextInput, username: str = Depends(get_optional_user)):
    """Permite processamento com ou sem autenticação (modo visitante)"""
    try:
        print(f"\n{'='*60}")
        print(f"📝 Processando texto ({len(data.text)} caracteres)")
        print(f"👤 Usuário: {username}")
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
async def shuffle_questions(data: ShuffleInput, username: str = Depends(verify_token)):
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

# =========================================================
# INICIALIZAÇÃO DO SERVIDOR
# =========================================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    
    print("\n" + "="*60)
    print("🎮 SchoolQuest API v4.0.0")
    print("="*60)
    print(f"🤖 Provedor de IA: OPENAI")
    print(f"📦 Modelo ativo: {MODEL_NAME}")
    print(f"🔐 Autenticação: Habilitada")
    print(f"👻 Modo Visitante: Habilitado (process-text)")
    print("💾 Cache: Ativado (24 horas)")
    print("🔒 CORS: Habilitado")
    print("="*60)
    print(f"📡 Servidor: http://0.0.0.0:{port}")
    print("🏠 Página inicial: / → redireciona para /login.html")
    print("📘 Documentação API: /docs")
    print("🏥 Health check: /api/health")
    print("="*60 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )