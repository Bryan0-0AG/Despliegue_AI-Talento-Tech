from groq import Groq
import os
from dotenv import load_dotenv

# Asegurar que cargue el archivo .env desde la raiz del proyecto y sobrescriba variables del sistema
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
dotenv_path = os.path.join(base_dir, '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# Inicializar el cliente de Groq con la clave de API
API_KEY = os.getenv("GROQ_API_KEY")
client = None
if API_KEY:
    API_KEY = API_KEY.strip().replace('"', '').replace("'", "")
    client = Groq(api_key=API_KEY)

def consultar_groq(mensaje, contexto_herramienta):
    # Validar que el cliente de Groq este correctamente inicializado
    if not client:
        return "⚠️ Error: API KEY de Groq no configurada en el archivo .env"

    try:
        # Definir el prompt del sistema y de usuario para el asistente
        prompt = f"""
        Eres un asistente experto integrado en SolarAI, una herramienta profesional desplegada para el analisis y prediccion de ahorro solar.
        CONTEXTO DE LA HERRAMIENTA: {contexto_herramienta}
        REGLAS:
        1. Responde de forma profesional, tecnica y formal, siendo claro y preciso en tus explicaciones cientificas y de ingenieria.
        2. Actua y habla de SolarAI como un producto comercial y una herramienta ya desplegada en produccion, nunca como un proyecto academico.
        3. Si la pregunta no es sobre la herramienta o energia solar, pide amablemente volver al tema de conversacion.
        4. Usa el contexto para responder sobre datos, modelos o inteligencia artificial.
        
        PREGUNTA: {mensaje}
        """
        
        # Enviar solicitud a la API de Groq usando Llama-3.3-70b
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error con Groq: {e}"

def responder_pregunta(mensaje, historial, df):
    mensaje_low = mensaje.lower()
    
    # Contexto del producto en produccion
    contexto = """
    Herramienta: SolarAI - Analizador de Ahorro Energetico Solar en Pereira.
    Estado: Sistema de produccion desplegado.
    Modelo: Random Forest Regressor (R2 Score de 97% en validacion).
    Variables analizadas: Año de instalacion, Tipo de instalacion, Material de paneles, Cantidad de paneles, Radiacion solar del area, Eficiencia, Humedad, Temperatura.
    """
    
    # Manejar saludo inicial formal y sin tutear
    if "hola" in mensaje_low:
        return "Hola. ¿En que puedo ayudarle con respecto a nuestra herramienta de analisis de energia solar?"
    
    return consultar_groq(mensaje, contexto)
