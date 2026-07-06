# Fuente de datos

El conjunto de datos proviene del trabajo de investigación  _Efecto del ultrasonido de intensidad alta y la fermentación sobre metabolitos específicos y propiedades funcionales del kéfir de agua_ {cite}`proenza2025`, en el cual se detalla el proceso experimental realizado con gránulos de kéfir de agua:

A una cantidad inicial de gránulos se le aplica un tratamiento de ultrasonido de intensidad y periodo de exposición definidos. Posteriormente, los gránulos se mantienen bajo condiciones controladas de temperatura y presión durante un periodo de 15 horas. Este proceso se repite hasta completarse en total un lapso de 175 horas en cuales se obtienen 15 puntos de medición. El dataset está compuesto por cinco series de tiempo, cada una conformada por 15 puntos equiespaciados. Cada serie representa el crecimiento de los gránulos asociado a un pretratamiento específico. En la siguiente figura se muestra una visualización de estas series, junto con una interpolación lineal entre los puntos medidos.

Los efectos que tiene el ultrasonido en el crecimiento de gránulos de kéfir es notorio al visualizar la serie de tiempo testigo ( sin tratamiento ) junto con las series con tratamiento . La serie de tiempo basal se nota que sigue el comportamiento esperado descrito en la literatura acerca de ello, es decir, sigue un modelo logístico simple.  A cambio, las series con mayor intensidad de ultrasonido parecen oscilar en vez de estabilizarse dentro del periodo de saturación . Esto implica una dinámica desconocida introducida por el tratamiento. 

Para lograr encontrar esta dinámica oculta, primero tenemos que partir de la ya conocida. Por ello, describiremos brevemente la formulación básica detrás de los modelos poblacionales que se utilizan para describir el crecimiento microbiano.  

```{figure} /images/raw_data_plot.png
:width: 72%
```


# Tratamiento de Datos

El procesamiento de los datos experimentales se realizó mediante un flujo automatizado de tipo ETL (Extract, Transform, Load), implementado en Python utilizando la librería pandas, con el objetivo de convertir la matriz de datos original en un formato analítico limpio y estructurado.

## Extracción
Los datos crudos fueron obtenidos a partir un archivo excel, correspondiente al registro de concentración de biomasa durante el proceso de fermentación de kéfir de agua bajo los distintos tratamientos de ultrasonido. Dado que el archivo original contenía encabezados y metadatos no relevantes para el análisis, se omitieron dichas filas, conservando únicamente la tabla de datos experimentales.

## Transformación
Una vez extraídos, los datos —originalmente en formato ancho, con una columna por cada tratamiento— fueron reestructurados a formato largo (long format) mediante la función `melt()`, generando las variables tratamiento y concentración (g/cm³). Previamente se eliminaron columnas sin contenido informativo (Unnamed), producto del formato original del archivo. Con el fin de vincular cada tratamiento con sus condiciones experimentales, se definió un diccionario de correspondencia que mapea la descripción textual de cada tratamiento (por ejemplo, "15 seg. 20 W/cm² (T2)") a un identificador estandarizado (tratamiento_1 a tratamiento_5). A partir de este identificador, un segundo diccionario permitió incorporar dos variables experimentales clave:

- intensidad(W/cm²)
- periodo de exposición(s)

De esta manera, cada observación quedó asociada no solo al tratamiento nominal, sino también a sus parámetros físicos de aplicación, lo que facilita el análisis del efecto de la intensidad y el tiempo de exposición sobre la concentración de biomasa a lo largo del tiempo de fermentación.

## Carga

Finalmente, el conjunto de datos transformado fue exportado en formato .csv, generando un archivo independiente por cada tratamiento (tratamiento_1.csv a tratamiento_5.csv), almacenados en el directorio data/processed/. Adicionalmente, se generó un archivo consolidado (control_dataset.csv) que reúne la totalidad de las observaciones, permitiendo tanto el análisis individual por tratamiento como el análisis comparativo global.
