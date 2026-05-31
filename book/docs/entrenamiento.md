# Construcción de la red neuronal

El tipo de red neuronal que se utiliza frecuentemente en el campo de aprendizaje 
automatizado científico suele ser una red neuronal profunda (Deep Neural Network, DNN)
de  2-3 capas ocultas unidireccionales de 50-100 neuronas . Las funciones de activación 
que  han mostrado buen desempeño en mayoría de casos de uso de problemas inversos con 
 ecuaciones diferenciales ordinarias han sido  $\tanh$, $\text{ReLu}$  ,$\text{Softmax}$ 
{cite}`Cuomo2022,Pappu2025,Farea2025` .

Después, se construye una red neuronal profunda que recibe como entrada el tiempo $t$ 
y produce como salida una aproximación continua de la población microbiana $\hat{u}(t)$. 
En el caso del problema inverso, los parámetros biológicos desconocidos del modelo de 
crecimiento se incorporan explícitamente como variables entrenables adicionales. Mediante 
diferenciación automática, se calculan las derivadas temporales de la salida de la red, 
necesarias para evaluar el residuo de la ecuación diferencial que gobierna el crecimiento 
microbiano. 

La arquitectura de esta red neuronal consiste de:
- **Capa de entrada**: Capa de una sola neurona con función de activación $\sigma = \tanh$.

- **Capas ocultas**: 3 capas ocultas densas 50 neuronas cada una con función de activación $\sigma = \tanh$.

- **Capa de salida**: Capa de una sola neurona con función de activación $\sigma = \tanh$.

## Definición del modelo físico y dominio

Definimos el modelo de crecimiento poblacional que describe el
fenómeno de estudio, típicamente mediante ecuaciones diferenciales ordinarias de tipo 
logístico o Gompertz que constituyen la restricción principal que se impondrá durante el entrenamiento. Se define 
el dominio temporal del problema, correspondiente al intervalo experimental de 168 horas, 
junto con las condiciones iniciales asociadas a la concentración inicial de gránulos de 
kéfir.

## Optimización
El entrenamiento de la PINN se realiza mediante algoritmos de optimización basados en pasos de gradiente, ajustando tanto los pesos de la red como  los parámetros del modelo. 
Durante este proceso, se busca minimizar la función de pérdida global, equilibrando el 
ajuste a datos y el cumplimiento de las ecuaciones gobernantes.

La correcta ponderación de los términos de la pérdida es un aspecto clave del flujo de trabajo, ya que un desbalance puede afectar la estabilidad del entrenamiento o sesgar 
la inferencia de parámetros. En el caso del problema inverso, se utilizó el método de "adam" para manejar el descenso de gradiente. 

## Evaluación y análisis de resultados

Una vez entrenada la red, se evalúa su desempeño comparando la solución
reconstruida con los datos experimentales disponibles. Los resultados obtenidos con distintas configuraciones (tanto modelo 
matemático como red ) se comparan utilizando métricas de desempeño, lo que nos permitirá 
seleccionar el modelo más adecuado para describir la dinámica de crecimiento microbiano 
en función del tratamiento experimental.

Con el objetivo de evaluar y comparar los resultados obtenidos al resolver problemas inversos mediante distintas configuraciones de Redes Neuronales Informadas por Modelos Físicos (PINNs), se emplearon métricas estadísticas ampliamente utilizadas en el análisis de modelos. En particular, se utilizaran el Criterio de Información de Akaike (AIC), el Criterio de Información Bayesiano (BIC) y el coeficiente de determinación ($R^2$), las cuales permiten cuantificar tanto la calidad del ajuste a los datos experimentales como la complejidad del modelo inferido.

En el contexto de este proyecto, estas métricas se aplican para comparar diferentes PINNs utilizadas en la estimación de parámetros biológicos del modelo de crecimiento microbiano (por ejemplo, el modelo de Gompertz) a partir de las series de tiempo del crecimiento de gránulos de kéfir de agua sometidos a distintos pretratamientos de ultrasonido.

### Criterio de Información de Akaike (AIC)
El AIC evalúa el equilibrio entre la bondad del ajuste del modelo y su complejidad, penalizando explícitamente el número de parámetros estimados. En problemas inversos resueltos con PINNs, esta métrica resulta especialmente útil para comparar configuraciones que ajustan los parámetros del modelo de crecimiento bajo distintas arquitecturas de red, funciones de activación o estrategias de ponderación de la función de pérdida.
Un valor menor de AIC indica un modelo más parsimonioso, es decir, aquel que logra describir adecuadamente la dinámica de crecimiento microbiano del kéfir utilizando el menor número efectivo de parámetros. Por ello, las PINNs con valores de AIC más bajos se consideran preferibles desde un punto de vista estadístico.

### Criterio de Información Bayesiano (BIC)
El BIC es conceptualmente similar al AIC, pero introduce una penalización más severa por el número de parámetros, lo que lo hace particularmente adecuado en escenarios con conjuntos de datos pequeños, como las series de tiempo disponibles en este estudio. Esta métrica favorece modelos más simples y robustos, reduciendo el riesgo de sobreajuste al resolver problemas inversos.

En este trabajo, el BIC permite identificar qué configuraciones de PINNs logran capturar el crecimiento de los gránulos de kéfir de agua de manera consistente, manteniendo al mismo tiempo una estructura de modelo lo más sencilla posible. Al igual que en el caso del AIC, valores menores de BIC indican un mejor desempeño global.

### Coeficiente de determinación ($R^2$)
El coeficiente de determinación ($R^2$) mide la proporción de la variabilidad de los datos experimentales que es explicada por el modelo ajustado. Dentro del marco de las PINNs aplicadas a problemas inversos, esta métrica permite evaluar directamente la capacidad de la solución inferida para reproducir las curvas de crecimiento observadas en los gránulos de kéfir de agua.
Valores de $R^2$ cercanos a 1 indican un alto grado de concordancia entre las predicciones del modelo de crecimiento —con parámetros inferidos por la PINN— y las mediciones experimentales. No obstante, dado que esta métrica no penaliza explícitamente la complejidad del modelo, su interpretación debe complementarse con criterios de información como el AIC y el BIC.
