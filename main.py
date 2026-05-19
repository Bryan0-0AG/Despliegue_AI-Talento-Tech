import os
import subprocess
import sys
import warnings
# Desactivar advertencias para mantener la consola limpia y libre de ruido
warnings.filterwarnings("ignore")

from src.processing.cleaner import limpiar_datos
from src.ai.trainer import entrenar_modelo_clasificacion

def main():
    ruta_data = os.path.join("data", "energia_solar_pereira_colombia.xlsx")
    
    if not os.path.exists(ruta_data):
        print(f"Error: No se encontro el archivo en {ruta_data}")
        return

    print("--- [MODULO 1] LIMPIEZA Y ETIQUETADO ---")
    df_clean = limpiar_datos(ruta_data)

    if df_clean is not None:
        print("\n--- [MODULO 2] ENTRENAMIENTO CLASIFICADOR ---")
        entrenar_modelo_clasificacion(df_clean)
        
        print("\n--- [MODULO 3] LANZAMIENTO INTEGRADO ---")
        print("Cargando Interfaz con Dashboard 100% integrado...")
        
        from src.interfaces.gradio_app import build_app
        import gradio as gr
        
        # Construir y lanzar la aplicacion directamente
        demo, custom_css = build_app()
        print("\n¡Todo listo! Accede al link que aparece abajo:")
        demo.launch(share=True, theme=gr.themes.Soft(), css=custom_css)

if __name__ == "__main__":
    main()