import gradio as gr
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from src.chatbot.engine import responder_pregunta
from src.analysis.visualizer import (
    obtener_graficos_categoricos,
    obtener_graficos_numericos,
    obtener_graficos_analisis_target,
    obtener_graficos_correlacion,
    obtener_graficos_correlacion_especifica
)

# Cargar recursos globales una sola vez al inicio para optimizar velocidad
MODELO = joblib.load('models/modelo_solar.pkl')
PREPRO = joblib.load('models/preprocesador.pkl')
DF_SOLAR = pd.read_excel('data/energia_solar_pereira_colombia_clean.xlsx')
DF_IND = pd.read_excel(os.path.join("data", "colombia_indicadores_2018_2100.xlsx"))

def load_resources():
    return MODELO, PREPRO, DF_SOLAR

def predecir_ahorro(anio, tipo, material, paneles, radiacion, eficiencia, humedad, temperatura, factura_sin_solar):
    try:
        # Asegurar tipo de año entero para la busqueda
        anio = int(anio)
        
        # Buscar datos del año en los indicadores precargados
        datos_anio = DF_IND[DF_IND.iloc[:, 0].astype(int) == anio]
        if datos_anio.empty:
            datos_anio = DF_IND.tail(1)
            
        ipc_g = float(datos_anio['ipc_general_pct'].values[0])
        ipc_e = float(datos_anio['ipc_energia_pct'].values[0])
        trm = float(datos_anio['trm_promedio_cop'].values[0])

        # Crear DataFrame con todas las columnas que el modelo espera
        input_df = pd.DataFrame([[
            anio, tipo, material, paneles, radiacion, 
            eficiencia, humedad, temperatura,
            ipc_g, ipc_e, trm
        ]], columns=[
            'Año Instalación', 'Tipo', 'Material Panel', 'N° Paneles', 'Radiación Solar',
            'Eficiencia Panel (%)', 'Humedad Relativa Prom', 'Temperatura Prom',
            'ipc_general_pct', 'ipc_energia_pct', 'trm_promedio_cop'
        ])
        
        # Obtener prediccion base del modelo
        input_pre = PREPRO.transform(input_df)
        base_pred = MODELO.predict(input_pre)[0]

        # ----------------------------------------------------
        # AJUSTE FISICO-ECONOMICO INTELIGENTE
        # ----------------------------------------------------
        # 1. Radiacion solar (Peso protagonico en la produccion)
        factor_radiacion = float(radiacion) / 5.5
        
        # 2. Eficiencia del panel (A mayor eficiencia, mayor ahorro)
        factor_eficiencia = float(eficiencia) / 18.5
        
        # 3. Humedad relativa (A mayor humedad, menor ahorro por nubosidad)
        factor_humedad = 1.0 - ((float(humedad) - 75.0) / 100.0) * 0.4
        
        # 4. Temperatura promedio (La temperatura adecuada favorece la produccion)
        factor_temperatura = 1.0 + ((float(temperatura) - 22.0) / 50.0) * 0.2
        
        # 5. Efecto del año e inflacion de energia acumulada en el tiempo (Ahorro monetario en COP)
        factor_inflacion = (1.0 + ipc_e / 100.0) ** max(0, anio - 2024)

        # Calcular el valor final escalado
        valor_predicho = base_pred * factor_radiacion * factor_eficiencia * factor_humedad * factor_temperatura * factor_inflacion
        # ----------------------------------------------------

        # Calcular el porcentaje real de ahorro respecto a la factura sin solar
        # No se puede ahorrar mas del 100% de la factura original
        porcentaje = (valor_predicho / float(factura_sin_solar)) * 100
        porcentaje = min(porcentaje, 100.0)

        # Formatear el resultado con lenguaje amigable y llamativo
        explicacion = f"## 💰 Tu ahorro estimado: **${valor_predicho:,.2f} COP** mensuales\n"
        explicacion += f"Esto equivale al **{porcentaje:.1f}%** de tu consumo actual sin paneles (${factura_sin_solar:,.0f} COP).\n\n"
        explicacion += f"Este calculo considera la **inflacion de energia ({ipc_e}%)** y la **TRM (${trm:,.0f})** para el año {anio}.\n\n"
        
        # Añadir un indicador visual de calidad
        if porcentaje > 60.0:
            explicacion += "🌟 ¡Excelente! Este sistema solar cubre la mayor parte de tu consumo energetico."
        elif porcentaje > 30.0:
            explicacion += "✅ ¡Muy bien! Estas teniendo un ahorro y cubrimiento considerable."
        else:
            explicacion += "💡 Es un ahorro inicial excelente, ¡cada porcentaje cuenta!"

        if anio > 2025:
            explicacion += f"\n\n*Nota: Proyeccion basada en los ultimos indicadores economicos disponibles.*"

        # Creamos un Medidor (Gauge) visual
        fig, ax = plt.subplots(figsize=(6, 2))
        
        # Dibujar barra de fondo y barra de progreso
        ax.barh([0], [100], color='#eeeeee', height=0.4)
        
        # Ajustar colores segun porcentaje solicitado
        if porcentaje <= 15.0:
            color_bar = '#F44336' # Rojo
        elif porcentaje <= 40.0:
            color_bar = '#FFC107' # Amarillo
        else:
            color_bar = '#4CAF50' # Verde
            
        ax.barh([0], [porcentaje], color=color_bar, height=0.4)
        
        ax.set_xlim(0, 100)
        ax.set_axis_off()
        ax.text(0, 0.35, "Porcentaje de Ahorro en tu Factura", fontsize=12, fontweight='bold')
        ax.text(porcentaje, -0.35, f"{porcentaje:.1f}%", ha='center', fontweight='bold', color=color_bar)
        
        plt.tight_layout()
        return explicacion, fig
    except Exception as e:
        return f"⚠️ Error interno en el calculador de ahorro: {str(e)}", None

def build_app():
    plt.close('all') # Limpiar figuras previas para evitar warnings de memoria
    modelo, prepro, df = load_resources()

    custom_css = """
    #header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 30px;
        background: linear-gradient(135deg, #FFF9C4 0%, #FFF176 100%);
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    #header img {
        border-radius: 10px;
        max-height: 100px;
        width: auto !important;
    }
    #header h1 {
        margin: 0 !important;
        font-weight: 800;
        color: #F57F17;
        font-family: 'Outfit', sans-serif;
    }
    #title {
        display: flex;
        align-items: center;
        background: transparent;
    }
    """
    with gr.Blocks(title="SolarAI") as demo:
        with gr.Row(elem_id="header"):
            gr.Image("data/logo.png", height=50, show_label=False, container=False, interactive=False, show_download_button=False)
            gr.Markdown("# SolarAI", elem_id="title")

        with gr.Tabs():
            with gr.TabItem("💬 Preguntale a la IA"):
                gr.ChatInterface(fn=lambda m, h: responder_pregunta(m, h, df))

            with gr.TabItem("📊 Simular mi Ahorro"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🛠️ Datos de tu Instalacion")
                        anio = gr.Number(label="¿En que año proyectamos?", value=2024)
                        tipo = gr.Dropdown(choices=list(df['Tipo'].unique()), label="Tipo de instalacion")
                        material = gr.Dropdown(choices=list(df['Material Panel'].unique()), label="Material de los paneles")
                        paneles = gr.Slider(1, 100, label="Cantidad de paneles", value=10)
                        radiacion = gr.Slider(1, 10, label="Nivel de sol (Radiacion)", value=5.5)
                        factura_sin_solar = gr.Slider(100000, 3000000, step=50000, label="Factura mensual actual sin paneles (COP)", value=600000)
                        
                        with gr.Accordion("⚙️ Datos tecnicos avanzados", open=False):
                            eficiencia = gr.Slider(10, 25, label="Eficiencia del panel (%)", value=18.5)
                            humedad = gr.Slider(30, 90, label="Humedad relativa (%)", value=75.0)
                            temperatura = gr.Slider(15, 35, label="Temperatura promedio (°C)", value=22.0)
                        
                        btn = gr.Button("¡Predecir mi ahorro!", variant="primary")
                    with gr.Column():
                        gr.Markdown("### 📈 Resultado de tu ahorro")
                        res = gr.Markdown("Aqui aparecera tu resultado...")
                        plt_plot = gr.Plot()
                
                btn.click(
                    predecir_ahorro, 
                    inputs=[anio, tipo, material, paneles, radiacion, eficiencia, humedad, temperatura, factura_sin_solar], 
                    outputs=[res, plt_plot]
                )

            with gr.TabItem("📊 Dashboard de Analisis"):
                gr.Markdown("## 📈 Analisis de la Base de Datos Solar")
                
                with gr.Tabs():
                    with gr.Tab("General"):
                        gr.Markdown("### 🎯 Correlacion: Variables Clave vs Ahorro")
                        gr.Plot(value=obtener_graficos_correlacion_especifica(df))
                    
                    with gr.Tab("Distribuciones"):
                        cat_figs = obtener_graficos_categoricos(df)
                        num_figs = obtener_graficos_numericos(df)
                        
                        with gr.Tabs():
                            with gr.Tab("Técnicas / Instalación"):
                                gr.Markdown("### 🛠️ Variables de Instalación")
                                with gr.Row():
                                    if 'Año Instalación' in cat_figs: gr.Plot(value=cat_figs['Año Instalación'], label="Año de Instalación")
                                    if 'Material Panel' in cat_figs: gr.Plot(value=cat_figs['Material Panel'], label="Material del Panel")
                                    if 'Tipo' in cat_figs: gr.Plot(value=cat_figs['Tipo'], label="Tipo de Instalación")
                                with gr.Row():
                                    if 'N° Paneles' in num_figs: gr.Plot(value=num_figs['N° Paneles'], label="N° de Paneles")
                                    if 'Eficiencia Panel (%)' in num_figs: gr.Plot(value=num_figs['Eficiencia Panel (%)'], label="Eficiencia (%)")

                            with gr.Tab("Climatológicas"):
                                gr.Markdown("### ☁️ Variables del Clima")
                                with gr.Row():
                                    if 'Radiación Solar' in num_figs: gr.Plot(value=num_figs['Radiación Solar'], label="Radiación Solar")
                                    if 'Humedad Relativa Prom' in num_figs: gr.Plot(value=num_figs['Humedad Relativa Prom'], label="Humedad Relativa")
                                    if 'Temperatura Prom' in num_figs: gr.Plot(value=num_figs['Temperatura Prom'], label="Temperatura Promedio")

                            with gr.Tab("Económicas"):
                                gr.Markdown("### 💰 Variables Económicas y Ahorro")
                                with gr.Row():
                                    if 'ipc_energia_pct' in num_figs: gr.Plot(value=num_figs['ipc_energia_pct'], label="IPC Energía (%)")
                                    if 'trm_promedio_cop' in num_figs: gr.Plot(value=num_figs['trm_promedio_cop'], label="TRM Promedio")
                                    if 'Factura Ahorrada Mensual' in num_figs: gr.Plot(value=num_figs['Factura Ahorrada Mensual'], label="Ahorro Mensual (COP)")

            with gr.TabItem("📘 Acerca de"):
                gr.Markdown("### ☀️ Sobre SolarAI")
                gr.Markdown("""
                Esta aplicacion utiliza Inteligencia Artificial para predecir el ahorro economico de instalaciones solares en Pereira, Colombia.
                
                **Caracteristicas clave:**
                - **Chatbot Inteligente:** Resuelve tus dudas sobre energia solar.
                - **Simulador Avanzado:** Predicciones ajustadas a la inflacion (IPC) y TRM.
                - **Dashboard Integrado:** Visualizacion completa de los datos recolectados.
                
                *Desarrollado para el Diplomado en IA - TalentoTech.*
                """)

    return demo, custom_css

