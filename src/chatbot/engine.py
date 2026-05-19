from google import genai
import os
from dotenv import load_dotenv

# REQUISITO: Asegurar que cargue el .env desde la raiz del proyecto y sobrescriba variables del sistema
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
dotenv_path = os.path.join(base_dir, '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# Inicializar el cliente con el nuevo SDK
API_KEY = os.getenv("GEMINI_API_KEY")
client = None
if API_KEY:
    API_KEY = API_KEY.strip().replace('"', '').replace("'", "")
    client = genai.Client(api_key=API_KEY)

def consultar_gemini(mensaje, contexto_proyecto):
    if not client:
        return "⚠️ Error: API KEY no configurada en el archivo .env"

    try:
        # Usar el nuevo metodo de generacion del SDK v2.0
        prompt = f"""
        Eres un asistente experto en el proyecto 'Energia Solar Pereira'.
        CONTEXTO: {contexto_proyecto}
        REGLAS:
        1. Responde de forma profesional, tecnica y formal, siendo claro y preciso en tus explicaciones cientificas y de ingenieria.
        2. Si la pregunta no es sobre el proyecto, pide amablemente volver al tema de conversacion.
        3. Usa el contexto para responder sobre datos, modelos o inteligencia artificial.
        
        PREGUNTA: {mensaje}
        """
        
        # Intentar con el nombre mas compatible
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error con Gemini (v2): {e}"

def responder_pregunta(mensaje, historial, df):
    mensaje_low = mensaje.lower()
    contexto = """
    Proyecto: Clasificacion de ahorro solar en Pereira.
    Modelo: Random Forest (Accuracy 96.5%).
    Variables: Año, Tipo, Material, Paneles, Radiacion.
    """
    if "hola" in mensaje_low:
        return "Hola. ¿En que puedo ayudarle con respecto al proyecto de energia solar?"
    
    return consultar_gemini(mensaje, contexto)
