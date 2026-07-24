## Ajuste de modelo con datos de tratamientos de ultrasonido

En este trabajo se logró **corroborar la validez de los parámetros del modelo de Gompertz** para describir el crecimiento de los gránulos de kéfir de agua. Para ello, se empleó el método numérico de **Runge–Kutta de cuarto orden (RK4)**, el cual permitió resolver la ecuación diferencial asociada al modelo de manera precisa y estable.

Los resultados obtenidos mediante RK4 mostraron una **alta concordancia entre la solución numérica y los datos experimentales de crecimiento testigo** de nuestra fuente de datos, lo que confirma que los parámetros estimados del modelo de Gompertz representan adecuadamente la dinámica del sistema biológico estudiado. Esta corroboración respalda el uso del modelo como una herramienta confiable para describir el comportamiento temporal del crecimiento de los gránulos. 


## Ajuste de PINN
Adicionalmente, se logró **ajustar una Red Neuronal Informada por la Física (Physics-Informed Neural Network, PINN)** a los mismos datos experimentales de crecimiento, utilizando igualmente el **modelo logístico como restricción física**.


::::{grid} 2

:::{grid-item}
```{figure} /images/solution_plot.png
:width: 100%
```
:::

:::{grid-item}
```{figure} /images/adj_loss_plot.png
:width: 100%
```
:::

::::

| Parámetro | Estimación | Real | Error absoluto | Error relativo %|
|-----------|------------|------|----------------|-----------------|
|     r (1/hr)  |   0.0491   | 0.046|     0.0031     |      6.74       |
|     m (g/L)   |   46.70    | 47.81|       1.11     |      2.32       |


 
El PINN incorporó la ecuación diferencial del modelo dentro de su función de pérdida, permitiendo que la red aprendiera el comportamiento del sistema respetando las leyes que gobiernan su dinámica.

El ajuste mediante PINNs mostró desempeño consistente con el método numérico clásico, reproduciendo de forma adecuada la evolución del crecimiento de los gránulos de kéfir de agua. Estos resultados no ayudarán para la siguiente etapa del proyecto que consiste en resolver un problema inverso con los datos que tenemos a la mano.


## Dinámica aprendida

El mejor resultado del proceso de descubrimiento de física vino siendo producto del entrenamiento de PINN con la función de corrección de **intensidad por periodos**. Se puede visualizar el desempeño del modelo en la siguientes imagenes: 

```{figure} /images/best_disc_loss.png
:width: 72%
```

```{figure} /images/best_disc_results.png
:width: 72%
```

| Índice | Valor |Unidades|
|--------|-------|--------|
| 1 | 2.0700e-02 |(g/cm3)|
| 2 | 6.0500e-05 |(g/cm3)/(W/cm2) |
| 3 | -6.0200e-04 |(g/cm3)/(s) |
| 4 | 7.5400e-06 |(g/cm3)/(W/cm2)(s) |


| Métrica                               | Valor                                                                                                                                    |
|---------------------------------------|------|
| Root Mean Square Error (RMSE)         | 2.9640  | |
| Mean Absolute Error (MAE)             | 2.4397  | |
| Mean Absolute Percentage Error (MAPE) | 8.0278  | 
| Akaike Information Criterion (AIC)    | 10742.5966                           
| Bayesian Information Criterion (BIC)  | 14534.2054                                  
| Coeficiente de determinación ($R^2$)  |  0.8961  |


## Conclusiones

El modelo obtenido mediante la función de corrección de **intensidad por periodos** presentó el mejor desempeño entre las configuraciones evaluadas, con resultados que respaldan su validez tanto desde el punto de vista estadístico como físico.

En términos de ajuste, el coeficiente de determinación ($R^2 = 0.8961$) indica que el modelo explica una proporción considerable de la varianza observada en la concentración de biomasa, lo que sugiere una representación adecuada del fenómeno de fermentación bajo las condiciones experimentales consideradas. El error cuadrático medio (RMSE = 2.9640) y el error absoluto medio (MAE = 2.4397) muestran una desviación moderada respecto a los valores observados, mientras que el error porcentual absoluto medio (MAPE = 8.0278%) confirma que dicha desviación se mantiene dentro de un margen razonable en relación con la escala del problema {cite}`Lewis1982industrial`(ver p.40). 

Desde una perspectiva de interpretabilidad física, la superioridad del ajuste basado en intensidad por periodos resulta consistente con la naturaleza del proceso experimental. Este resultado es coherente con el diseño experimental, en el cual el sistema fue perturbado en intervalos de tiempo fijos, aplicando una intensidad y un periodo de exposición constantes en cada tratamiento. Al incorporar explícitamente esta interacción entre intensidad y periodo, el modelo refleja de manera más fiel el mecanismo mediante el cual la ultrasonicación afecta el crecimiento de los gránulos de kéfir, lo que explica su mejor desempeño frente a formulaciones alternativas.

Estos resultados no solo validan cuantitativamente el modelo propuesto, sino que también refuerzan la hipótesis de que la relación funcional entre intensidad y periodo de exposición constituye un factor determinante en la dinámica de fermentación observada, ofreciendo así una base sólida para la interpretación física del fenómeno estudiado.

Sin embargo,se tienen algunas limitaciones del presente estudio. En primer lugar, el modelo fue ajustado y validado sobre un rango acotado de combinaciones de intensidad y periodo de exposición, por lo que su capacidad de extrapolación a condiciones experimentales fuera de este rango no han sido evaluada. Asimismo, el tamaño de la muestra utilizada para el entrenamiento del modelo, si bien suficiente para obtener un ajuste estadísticamente aceptable, limita la robustez de las estimaciones de los coeficientes, particularmente en lo relativo a su significancia individual. 

Finalmente, aunque las métricas de error se encuentran dentro de umbrales razonables, no se descarta la posibilidad de cierto grado de sobreajuste dado el número de parámetros del modelo en relación con la cantidad de observaciones disponibles. Futuros trabajos podrían abordar estas limitaciones mediante la incorporación de un conjunto de datos más amplio y diverso, así como la validación cruzada del modelo bajo condiciones experimentales adicionales.