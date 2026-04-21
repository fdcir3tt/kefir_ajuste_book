# Descubrimiento de dinámicas ocultas con PINNs


Contamos con datos que describen con buena precisión el fenómeno bien conocido: el crecimiento logístico de los granos de kéfir en base lactosa. Asimismo, existe evidencia razonable para suponer que los granos de kéfir en base acuosa siguen un comportamiento similar.
Al analizar los datos bajo ciertas condiciones de tratamiento, se observa una desviación respecto a la trayectoria esperada. Dado que esta diferencia no es drástica, se parte del modelo logístico clásico como base, incorporando un término adicional desconocido que representa la “nueva física” que se busca descubrir.

# PINNs: Una herramienta imprevista 
Las redes neuronales informadas por física (PINNs) permiten integrar conocimiento previo directamente en el proceso de entrenamiento mediante restricciones basadas en ecuaciones diferenciales. Esta característica no solo mejora la consistencia del modelo, sino que también abre la posibilidad de aprender dinámicas ocultas presentes en los datos.
Bajo este enfoque, se asume que un conjunto de datos $X=\{C_n, \mathcal{D}\}$ sigue una dinámica conocida con pequeñas desviaciones. Esto se modela como:

$$
 \frac{dP}{dt} = \mathcal{F}(t,P;\lambda) +\delta(t,P,I,T;c_j)
$$

donde el término $\delta$ representa la contribución desconocida que se desea identificar.
En este caso particular, la desviación depende de variables de control $I$ y $T$ lo que requiere adaptar tanto la arquitectura de la red como la estructura del conjunto de datos para capturar adecuadamente estas dependencias.



# Dinámicas propuestas
Para modelar la dinámica oculta \delta, se consideraron distintas aproximaciones:
- **Modelos multipolinomiales**: donde \delta se expresa como una combinación de términos polinomiales con coeficientes ajustables.

$$
\delta(I,T;c_j) = \sum_{j=0}^{N}c_j I^{a_j} T^{b_j},
$$


-  **Modelos de Intensidad**: que representan la dinámica mediante sumas de funciones sinusoidales, caracterizadas por amplitudes y frecuencias.

$$
\delta(t,I,T;c_j) = (c_1 + c_2I+c_3T+c_4IT)*sin(\frac{2\pi}{15}t),
$$

- **Modelo de red + regresión**: que combinan una red neuronal con una función de corrección explícita, buscando capturar tanto patrones conocidos como comportamientos no estructurados.
$$
\delta(t;c_j=\theta) = NN_\theta(t),
$$

# Implementación 
La implementación se basa en el entrenamiento de una PINN que incorpora tanto el modelo logístico como el término correctivo $\delta$. Durante el entrenamiento, se optimizan simultáneamente los parámetros del modelo base y la forma funcional de la dinámica oculta. De esta forma, la función de perdida queda de la forma:

$$
 \mathcal{L}(\theta,c_j;X)=\sum_{t_i \in C_n}\frac{1}{2}f(\hat P_i,t_i;c_j)^2+\sum_{t_i\in \mathcal{D}}\frac{1}{2}(P_i-\hat P_i)^2,
$$
donde la función de residuos es:
$$
f(\hat P,t;c_j) = \frac{d\hat P}{dt} -\mathcal{F}(\hat P,t;\lambda)-\delta(t,P,I,T;c_j) ,
$$




