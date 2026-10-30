Requisitos

- Python 3 con numpy, scipy, matplotlib y ffmpeg para videos .mp4.

Uso

Abrir una terminal en la carpeta del código y reemplazar las palabras en mayúscula
```
python plantilla_s7.py --minutos MINUTOS --iters ITERACIONES --m_max NÚMERO MÁXIMO DE GRUPOS --eps COTA EPSILON --escala_min ESCALA MÍNIMA --procesos CANTIDAD DE PROCESOS --CARPETA DE GUARDADO
```
Dejamos un ejemplo
```
python plantilla_s7.py --minutos 480 --iters 3000 --m_max 9 --eps 1e-9 --escala_min -8 --procesos 12 --carpeta noche_s7
```

Refinamiento en alta precisión de las mejores plantillas: Vuelve a iterar con más pasos
```
python plantilla_s7.py refinar --carpeta CARPETA CON RESULTADOS --top CANTIDAD DE PLANTILLAS
```
Un ejemplo
```
python plantilla_s7.py refinar --carpeta noche_s7 --top 5
```


Plots y videos
```
python dibujar_grafo.py noche_s7 --mejor                     # imagen de la mejor plantilla
python dibujar_grafo.py noche_s7 --mejor --video             # video de su búsqueda
python dibujar_grafo.py noche_s7 --mejor --video --cadena    # video de toda la cadena de búsquedas
```
