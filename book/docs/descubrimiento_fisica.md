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
- **Modelos multipolinomiales**: donde $\delta$ se expresa como una combinación de términos polinomiales con coeficientes ajustables.

$$
\delta(I,T;c_j) = \sum_{j=0}^{N}c_j I^{a_j} T^{b_j},
$$


-  **Modelos de Intensidad**: $\delta$ se representa por la multiplicación de un término de interacción entre la intensidad $I$ y periodo de exposición $T$ por un termino senoidal tomando en cuenta el periodo de aplicación de tratamiento (15 horas).

$$
\delta(t,I,T;c_j) = (c_1 + c_2I+c_3T+c_4IT)*sin(\frac{2\pi}{15}t),
$$

- **Modelo de red + regresión**: que combinan una red neuronal buscando capturar comportamientos no estructurados y después extraer patrones esperados con un método de regresión.
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
f(\hat P,t;c_j) = \frac{d\hat P}{dt} -\mathcal{F}(\hat P,t;\lambda)-\delta(t,P,I,T;c_j) .
$$

Primero, cargamos nuestros datos de entrada `(t,I,T)`: 
```python
    
t0,y0 = load_initial_conditions(dataset)
t0,tf = load_time_domain(dataset)

X_train, y_train, X_test,y_test = split_train_data(dataset)
```
Y se define el método `ode` que nos sirve como auxiliar de la función de residuos $f(\hat P,t;c_j)$:
```python    

def ode(x, y):
    I_t = x[:, 0:1]
    T_t = x[:, 1:2]
    t = x[:, 2:3]

    delta = correction_function(I_t, T_t, c_coef,t,**kwargs)

    return equation(x,y)- delta
```

Definimos el dominio de datos que se utilizarán para el entrenamiento de nuestra red nueva:

```python  
t_min, t_max = float(t0), float(tf)
I_min, I_max = X_train[:, 0].min(), X_train[:, 0].max()
T_min, T_max = X_train[:, 1].min(), X_train[:, 1].max()
    
geom_space = dde.geometry.Rectangle([I_min, T_min],
                                    [I_max, T_max])

timedomain = dde.geometry.TimeDomain(t_min, t_max)
geom = dde.geometry.GeometryXTime(geom_space, timedomain)

#  >Método de colocación<

observe_bc = dde.icbc.PointSetBC(anchor_X.astype(np.float32),
                                 observe_y.astype(np.float32),
                                 component=0,
                                 shuffle=False)
data_pinn = dde.data.PDE(geometry=geom,
                         pde=ode,
                         bcs=[observe_bc],
                         num_domain=200,       
                         num_boundary=0,
                         anchors=anchor_X.astype(np.float32))
```
y cambiamos la capa de entrada para que pueda recibir datos de forma `(t,I,T)`

```python
net = dde.nn.FNN([3, 50, 50, 50, 1], "tanh", "Glorot uniform")

```


# Resultados