# Guía para Usar Modelos ONNX

Esta guía explica cómo configurar y usar tus modelos ONNX en el servidor.

## 📦 Instalación

Primero, instala las dependencias (incluye `onnxruntime-node`):

```bash
npm install
```

## 📁 Estructura de Carpetas

Coloca tus modelos ONNX en la carpeta `models/` en la raíz del proyecto:

```
ml_server/
├── models/
│   ├── churn_knn.onnx
│   ├── churn_logreg.onnx
│   └── credit_kmeans.onnx
├── src/
└── ...
```

## ⚙️ Configuración

### Opción 1: Variables de Entorno (Recomendado)

Crea o edita el archivo `.env`:

```env
PORT=3000
NODE_ENV=development
CORS_ORIGIN=http://localhost:5173

# Rutas a los modelos ONNX
CHURN_KNN_MODEL_PATH=models/churn_knn.onnx
CHURN_LOGREG_MODEL_PATH=models/churn_logreg.onnx
CREDIT_KMEANS_MODEL_PATH=models/credit_kmeans.onnx
```

### Opción 2: Valores por Defecto

Si no defines las variables de entorno, el sistema buscará los modelos en:
- `models/churn_knn.onnx`
- `models/churn_logreg.onnx`
- `models/credit_kmeans.onnx`

## 📝 Formato de Entrada de los Modelos

### Modelos de Churn (KNN y Regresión Logística)

Los modelos deben esperar un tensor de entrada con **21 características** usando codificación one-hot (one-hot encoding).

**Forma del tensor**: `[1, 21]` (1 ejemplo, 21 características)

**Orden de las características**:

1. `adulto_mayor` (número, escalado)
2. `meses_como_cliente` (número, escalado)
3. `cargo_mensual` (número, escalado)
4. `tiene_pareja_Yes` (0 o 1)
5. `dependientes_Yes` (0 o 1)
6. `tipo_internet_Fiber optic` (0 o 1)
7. `tipo_internet_No` (0 o 1)
8. `seguridad_en_linea_No internet service` (0 o 1)
9. `seguridad_en_linea_Yes` (0 o 1)
10. `respaldo_en_linea_No internet service` (0 o 1)
11. `respaldo_en_linea_Yes` (0 o 1)
12. `proteccion_dispositivo_No internet service` (0 o 1)
13. `proteccion_dispositivo_Yes` (0 o 1)
14. `soporte_tecnico_No internet service` (0 o 1)
15. `soporte_tecnico_Yes` (0 o 1)
16. `tipo_contrato_One year` (0 o 1)
17. `tipo_contrato_Two year` (0 o 1)
18. `facturacion_electronica_Yes` (0 o 1)
19. `metodo_pago_Credit card (automatic)` (0 o 1)
20. `metodo_pago_Electronic check` (0 o 1)
21. `metodo_pago_Mailed check` (0 o 1)

**Nota sobre codificación one-hot**:
- Las categorías de referencia no tienen columna propia (todas quedan en 0)
- Por ejemplo, si `tipo_contrato` es "Month-to-month", las columnas 16 y 17 quedan en 0
- Si `metodo_pago` es "Bank transfer (automatic)", las columnas 19, 20 y 21 quedan en 0

**Formato de salida esperado**:
Los modelos de churn deben devolver dos salidas:
- **Salida 1 (label)**: La clase predicha `[0]` o `[1]` (0 = no abandona, 1 = abandona)
- **Salida 2 (probabilities)**: Probabilidades `[prob_no_churn, prob_churn]`

### Modelo K-Means (Tarjeta de Crédito)

El modelo debe esperar un tensor de entrada con **16 características** normalizadas usando StandardScaler.

**Forma del tensor**: `[1, 16]` (1 ejemplo, 16 características)

**Orden de las características** (ya normalizadas):

1. `Saldo` (normalizado)
2. `Frecuencia_Saldo` (normalizado)
3. `Compras_Totales` (normalizado)
4. `Compras_Contado` (normalizado)
5. `Compras_Cuotas` (normalizado)
6. `Avances_Efectivo` (normalizado)
7. `Frecuencia_Compras` (normalizado)
8. `Frec_Compras_Contado` (normalizado)
9. `Frec_Compras_Cuotas` (normalizado)
10. `Frec_Avances` (normalizado)
11. `Transacciones_Avance` (normalizado)
12. `Transacciones_Compra` (normalizado)
13. `Limite_Credito` (normalizado)
14. `Pagos_Realizados` (normalizado)
15. `Pago_Minimo` (normalizado)
16. `Pct_Pago_Completo` (normalizado)

**Nota**: El preprocesamiento aplica StandardScaler automáticamente usando los parámetros `mean` y `scale` almacenados en `preprocessing.ts`.

**Formato de salida esperado**:
- Un solo valor `[cluster_index]` (número entero, típicamente 0-4)

## 🔧 Ajustar el Código para Tu Modelo

Si tu modelo tiene un formato diferente, puedes ajustar:

### 1. Preprocesamiento (`src/utils/preprocessing.ts`)

Modifica las funciones `preprocessChurnData()` o `preprocessCreditData()` si necesitas:
- Diferente orden de características
- Diferente codificación de variables categóricas
- Normalización o escalado diferente

**Para Churn**: La función `preprocessChurnData()` aplica:
- Escalado MinMax a las variables numéricas (`adulto_mayor`, `meses_como_cliente`, `cargo_mensual`)
- Codificación one-hot a todas las variables categóricas

**Para Credit**: La función `preprocessCreditData()` aplica:
- StandardScaler usando los parámetros `mean` y `scale` almacenados

### 2. Procesamiento de Salida (en los servicios)

En `src/services/churnKnn.service.ts`, `churnLogReg.service.ts` o `creditKmeans.service.ts`, ajusta la sección que procesa la salida del modelo:

**Para Churn**:
```typescript
// El modelo debe tener 2 salidas: label y probabilities
const [labelName, probName] = this.modelSession.outputNames;
const labelTensor = results[labelName];
const probTensor = results[probName];

const prediccion = Number((labelTensor.data as any)[0]) as 0 | 1;
const probs = Array.from(probTensor.data as any).map(Number);
```

**Para Credit**:
```typescript
// El modelo devuelve el índice del cluster
const output = results[this.modelSession.outputNames[0]].data as any;
const cluster = typeof raw === 'bigint' ? Number(raw) : Math.round(Number(raw));
```

### 3. Nombres de Input/Output

El código usa automáticamente los primeros nombres de input/output del modelo:
- `this.modelSession.inputNames[0]`
- `this.modelSession.outputNames[0]` (y `[1]` para modelos de churn)

Si tu modelo tiene nombres específicos, puedes cambiarlos en los servicios.

## 🧪 Probar con Modelos ONNX

1. Coloca tus modelos en la carpeta `models/`
2. Inicia el servidor: `npm run dev`
3. Verás en la consola si los modelos se cargaron correctamente:
   - ✅ `Usando modelo ONNX para Churn KNN`
   - ✅ `Usando modelo ONNX para Churn Regresión Logística`
   - ✅ `Usando modelo ONNX para Credit K-Means`
   - ⚠️ `Modelo ONNX no encontrado...` (si falta algún modelo)

## 📚 Exportar Modelos a ONNX

### Desde Python (scikit-learn)

**Para modelos de Churn (KNN o LogisticRegression)**:

```python
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Tu modelo entrenado
model = ...  # tu modelo KNN o LogisticRegression

# Definir el tipo de entrada (21 características)
initial_type = [('float_input', FloatTensorType([None, 21]))]

# Convertir a ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Guardar
with open("churn_knn.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

**Para modelo K-Means de Credit**:

```python
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Tu modelo K-Means entrenado
model = ...  # tu modelo KMeans

# Definir el tipo de entrada (16 características)
initial_type = [('float_input', FloatTensorType([None, 16]))]

# Convertir a ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Guardar
with open("credit_kmeans.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

### Desde Python (PyTorch)

```python
import torch
import torch.onnx

# Tu modelo entrenado
model = ...
model.eval()

# Ejemplo de entrada
dummy_input = torch.randn(1, num_features)  # 21 para churn, 16 para credit

# Exportar
torch.onnx.export(
    model,
    dummy_input,
    "modelo.onnx",
    input_names=['input'],
    output_names=['output'],  # Para churn: ['label', 'probabilities']
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

## 🐛 Solución de Problemas

### Error: "Modelo ONNX no encontrado"
- Verifica que el archivo existe en la ruta especificada
- Verifica la ruta en `.env` o usa la ruta por defecto

### Error: "Error al cargar modelo ONNX"
- Verifica que el archivo ONNX no está corrupto
- Verifica que tienes `onnxruntime-node` instalado: `npm install onnxruntime-node`

### Error: "Shape mismatch" o "Input name mismatch"
- Verifica que el formato de entrada coincide con lo esperado
- Para churn: debe ser `[1, 21]`
- Para credit: debe ser `[1, 16]`
- Revisa los nombres de input/output del modelo
- Ajusta el preprocesamiento si es necesario

### Error: "Got invalid dimensions for input"
- Verifica que el número de características coincide
- Para churn: 21 características
- Para credit: 16 características

### El modelo se carga pero da resultados incorrectos
- Verifica que el preprocesamiento coincide con cómo entrenaste el modelo
- Verifica el orden de las características
- Verifica la codificación one-hot para modelos de churn
- Verifica que el StandardScaler usa los mismos parámetros para credit

### Error: "Non tensor type is temporarily not supported"
- Asegúrate de que los datos se convierten correctamente a `Float32Array`
- Verifica que el tensor se crea con la forma correcta

## 📞 Notas Adicionales

- Los modelos se cargan una vez al iniciar el servidor y se cachean
- Si cambias un modelo, reinicia el servidor
- El sistema usa CPU por defecto. Para usar GPU, cambia `'cpu'` a `'cuda'` en `onnxLoader.ts` (requiere onnxruntime-gpu)
- Los modelos de churn deben tener 2 salidas: `label` y `probabilities`
- El modelo de credit debe devolver el índice del cluster como un número entero

