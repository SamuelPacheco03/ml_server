# Guía para Usar Modelos ONNX

Esta guía explica cómo configurar y usar tus modelos ONNX reales en lugar de la simulación.

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

# Si es false, no usará simulación como fallback (solo ONNX)
USE_FALLBACK=true
```

### Opción 2: Valores por Defecto

Si no defines las variables de entorno, el sistema buscará los modelos en:
- `models/churn_knn.onnx`
- `models/churn_logreg.onnx`
- `models/credit_kmeans.onnx`

## 🔄 Comportamiento

El sistema funciona de la siguiente manera:

1. **Si el modelo ONNX existe**: Lo carga y lo usa para las predicciones
2. **Si el modelo ONNX NO existe**: Usa la simulación como fallback (si `USE_FALLBACK=true`)
3. **Si hay un error con ONNX**: Usa la simulación como fallback (si `USE_FALLBACK=true`)

## 📝 Formato de Entrada de los Modelos

### Modelos de Churn (KNN y Regresión Logística)

Los modelos deben esperar un tensor de entrada con **13 características** en este orden:

1. `adulto_mayor` (0 o 1)
2. `meses_como_cliente` (número)
3. `cargo_mensual` (número)
4. `tiene_pareja` (0 o 1, Yes=1, No=0)
5. `dependientes` (0 o 1, Yes=1, No=0)
6. `tipo_internet` (0=No, 1=DSL, 2=Fiber optic)
7. `seguridad_en_linea` (0 o 1, Yes=1, No=0)
8. `respaldo_en_linea` (0 o 1, Yes=1, No=0)
9. `proteccion_dispositivo` (0 o 1, Yes=1, No=0)
10. `soporte_tecnico` (0 o 1, Yes=1, No=0)
11. `tipo_contrato` (0=Month-to-month, 1=One year, 2=Two year)
12. `facturacion_electronica` (0 o 1, Yes=1, No=0)
13. `metodo_pago` (0=Electronic check, 1=Mailed check, 2=Bank transfer, 3=Credit card)

**Forma del tensor**: `[1, 13]` (1 ejemplo, 13 características)

**Formato de salida esperado**:
- Opción 1: Un solo valor `[probabilidad_churn]` (0-1)
- Opción 2: Dos valores `[prob_no_churn, prob_churn]`
- Opción 3: Un solo valor `[clase_predicha]` (0 o 1)

### Modelo K-Means (Tarjeta de Crédito)

El modelo debe esperar un tensor de entrada con **7 características** en este orden:

1. `BALANCE` (número)
2. `PURCHASES_FREQUENCY` (número, 0-1)
3. `CASH_ADVANCE` (número)
4. `PAYMENTS` (número)
5. `MINIMUM_PAYMENTS` (número)
6. `PRC_FULL_PAYMENT` (número, 0-1)
7. `CREDIT_LIMIT` (número)

**Forma del tensor**: `[1, 7]` (1 ejemplo, 7 características)

**Formato de salida esperado**:
- Un solo valor `[cluster_index]` (número entero, típicamente 0-4)

## 🔧 Ajustar el Código para Tu Modelo

Si tu modelo tiene un formato diferente, puedes ajustar:

### 1. Preprocesamiento (`src/utils/preprocessing.ts`)

Modifica las funciones `preprocessChurnData()` o `preprocessCreditData()` si necesitas:
- Diferente orden de características
- Diferente codificación de variables categóricas
- Normalización o escalado diferente

### 2. Procesamiento de Salida (en los servicios)

En `src/services/churnKnn.service.ts`, `churnLogReg.service.ts` o `creditKmeans.service.ts`, ajusta la sección que procesa la salida del modelo:

```typescript
// En predictWithOnnx(), ajusta según tu modelo:
const outputData = output.data as Float32Array;

// Si tu modelo devuelve algo diferente, modifica esta parte
```

### 3. Nombres de Input/Output

El código usa automáticamente los primeros nombres de input/output del modelo:
- `this.modelSession.inputNames[0]`
- `this.modelSession.outputNames[0]`

Si tu modelo tiene nombres específicos, puedes cambiarlos en los servicios.

## 🧪 Probar con Modelos ONNX

1. Coloca tus modelos en la carpeta `models/`
2. Inicia el servidor: `npm run dev`
3. Verás en la consola si los modelos se cargaron correctamente:
   - ✅ `Usando modelo ONNX para Churn KNN`
   - ⚠️ `Modelo ONNX no encontrado... Usando simulación`

## 📚 Exportar Modelos a ONNX

### Desde Python (scikit-learn)

```python
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Tu modelo entrenado
model = ...  # tu modelo KNN, LogisticRegression, o KMeans

# Definir el tipo de entrada
initial_type = [('float_input', FloatTensorType([None, num_features]))]

# Convertir a ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Guardar
with open("modelo.onnx", "wb") as f:
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
dummy_input = torch.randn(1, num_features)

# Exportar
torch.onnx.export(
    model,
    dummy_input,
    "modelo.onnx",
    input_names=['input'],
    output_names=['output'],
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
- Revisa los nombres de input/output del modelo
- Ajusta el preprocesamiento si es necesario

### El modelo se carga pero da resultados incorrectos
- Verifica que el preprocesamiento coincide con cómo entrenaste el modelo
- Verifica el orden de las características
- Verifica la codificación de variables categóricas

## 📞 Notas Adicionales

- Los modelos se cargan una vez al iniciar el servidor y se cachean
- Si cambias un modelo, reinicia el servidor
- El sistema usa CPU por defecto. Para usar GPU, cambia `'cpu'` a `'cuda'` en `onnxLoader.ts` (requiere onnxruntime-gpu)

