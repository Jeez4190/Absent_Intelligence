import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


def explicar_como_experto(datos):
    """
    Pide a Gemini una explicación sencilla, como si un experto
    se lo contara a alguien sin conocimientos técnicos.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Falta la clave de Gemini. Crea un archivo .env con "
            "GEMINI_API_KEY=tu_clave (la obtienes en https://aistudio.google.com/apikey)."
        )

    modelo = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    cliente = genai.Client(api_key=api_key)
    respuesta = cliente.models.generate_content(
        model=modelo,
        contents=_armar_prompt(datos),
    )

    texto = (respuesta.text or "").strip()
    if not texto:
        raise ValueError("Gemini no devolvió una explicación.")
    return texto


def _armar_prompt(datos):
    algoritmo = "árbol de decisión ID3" if datos.get("algoritmo") == "id3" else "vecinos más cercanos (K-NN)"
    valores = "\n".join(
        f"- {nombre}: {valor}"
        for nombre, valor in datos.get("valores", {}).items()
    )
    razones = "\n".join(f"- {razon}" for razon in datos.get("razones", [])) or "- Sin rangos estadísticos."
    pasos = "\n".join(f"- {paso}" for paso in datos.get("pasos_modelo", [])) or "- No aplica."
    vecinos = _texto_vecinos(datos.get("vecinos", []))
    probabilidades = "\n".join(
        f"- {item['clase']}: {item['probabilidad'] * 100:.1f}%"
        for item in datos.get("probabilidades", [])
    ) or "- No disponible."

    return f"""
Eres un experto en botánica y en inteligencia artificial, pero estás
explicando a una persona común, sin jerga innecesaria.

Redacta en español una explicación clara de por qué esta planta
pertenece a la especie predicha. Tono cercano, paciente y concreto.
Usa 2 o 3 párrafos cortos. No uses listas con viñetas. No inventes
números distintos a los que te doy. No menciones "prompt" ni que eres
un modelo de lenguaje.

Datos reales del caso:
- Especie predicha: {datos.get('prediccion')}
- Algoritmo usado: {algoritmo}
- Medidas ingresadas:
{valores}
- Confianza del modelo:
{probabilidades}
- Comparación con plantas de esa especie:
{razones}
- Cómo decidió el modelo:
{pasos}
{vecinos}

La explicación debe responder: qué planta es, por qué esas medidas
encajan con esa especie, y cómo el modelo llegó a esa conclusión
de forma fácil de entender.
""".strip()


def _texto_vecinos(vecinos):
    if not vecinos:
        return ""

    lineas = ["- Ejemplos de entrenamiento más parecidos:"]
    for i, vecino in enumerate(vecinos, start=1):
        medidas = ", ".join(
            f"{nombre}={valor}"
            for nombre, valor in vecino.get("valores", {}).items()
        )
        lineas.append(
            f"  {i}. clase {vecino.get('clase')}, "
            f"distancia {vecino.get('distancia'):.3f}, {medidas}"
        )
    return "\n".join(lineas)
