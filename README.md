# ML Server - API Backend para Modelos de Machine Learning

API backend completa construida con Node.js, Express y TypeScript para ejecutar modelos de machine learning ONNX en producción.

## 🚀 Características

- **Arquitectura en capas**: Routes, Controllers, Services
- **TypeScript**: Código tipado y seguro
- **Validación de datos**: Usando Zod para validar payloads
- **Manejo de errores centralizado**: Middlewares para errores y rutas no encontradas
- **CORS configurado**: Listo para consumir desde frontend React
- **Modelos ML con ONNX**:
  - KNN para predicción de churn
  - Regresión Logística para predicción de churn
  - K-Means para segmentación de clientes de tarjeta de crédito

## 📁 Estructura del Proyecto

```
ml_server/
├── models/                  # Coloca tus modelos ONNX aquí
│   ├── churn_knn.onnx
│   ├── churn_logreg.onnx
│   └── credit_kmeans.onnx
├── src/
│   ├── app.ts              # Configuración de Express
│   ├── server.ts            # Arranque del servidor
│   ├── config/
│   │   └── env.ts          # Variables de entorno
│   ├── controllers/
│   │   ├── churn.controller.ts
│   │   └── credit.controller.ts
│   ├── middlewares/
│   │   ├── errorHandler.ts
│   │   ├── notFoundHandler.ts
│   │   └── validateRequest.ts
│   ├── models/
│   │   └── types.ts        # Interfaces TypeScript
│   ├── routes/
│   │   ├── churn.routes.ts
│   │   └── credit.routes.ts
│   ├── services/
│   │   ├── churnKnn.service.ts
│   │   ├── churnLogReg.service.ts
│   │   └── creditKmeans.service.ts
│   └── utils/
│       ├── onnxLoader.ts   # Cargador de modelos ONNX
│       └── preprocessing.ts # Preprocesamiento de datos
├── package.json
├── tsconfig.json
└── .env                     # Variables de entorno (crear)
```

## 🛠️ Instalación y Configuración

### 1. Instalar Dependencias

```bash
npm install
```

### 2. Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
PORT=3000
NODE_ENV=development
CORS_ORIGIN=http://localhost:5173

# Rutas a los modelos ONNX (opcional, usa valores por defecto si no se especifican)
CHURN_KNN_MODEL_PATH=models/churn_knn.onnx
CHURN_LOGREG_MODEL_PATH=models/churn_logreg.onnx
CREDIT_KMEANS_MODEL_PATH=models/credit_kmeans.onnx
```

### 3. Colocar Modelos ONNX

Coloca tus modelos ONNX entrenados en la carpeta `models/`:

```
models/
├── churn_knn.onnx
├── churn_logreg.onnx
└── credit_kmeans.onnx
```

**Nota**: Si no colocas los modelos, el servidor iniciará pero los endpoints devolverán error. Ver `MODELOS_ONNX.md` para más detalles sobre el formato de los modelos.

### 4. Iniciar el Servidor

**Modo desarrollo** (con hot-reload):
```bash
npm run dev
```

**Compilar TypeScript**:
```bash
npm run build
```

**Modo producción**:
```bash
npm start
```

El servidor estará disponible en `http://localhost:3000` (o el puerto configurado en `.env`).

### 5. Verificar que Funciona

Abre otra terminal y prueba el endpoint de salud:

```bash
curl http://localhost:3000/health
```

Deberías recibir: `{"status":"ok"}`

## 📡 Endpoints

### Health Check
- **GET** `/health`
  - Respuesta: `{ "status": "ok" }`

### Churn - KNN
- **POST** `/api/churn/knn`
  - **Body** (JSON):
  ```json
  {
    "adulto_mayor": 0,
    "meses_como_cliente": 12,
    "cargo_mensual": 75.35,
    "tiene_pareja": "Yes",
    "dependientes": "No",
    "tipo_internet": "Fiber optic",
    "seguridad_en_linea": "No",
    "respaldo_en_linea": "No",
    "proteccion_dispositivo": "No",
    "soporte_tecnico": "No",
    "tipo_contrato": "Month-to-month",
    "facturacion_electronica": "Yes",
    "metodo_pago": "Electronic check"
  }
  ```
  - **Respuesta**:
  ```json
  {
    "model": "knn",
    "prediccion": 1,
    "probabilidad": 0.78,
    "mensaje": "El cliente tiene alta probabilidad de abandonar el servicio."
  }
  ```

### Churn - Regresión Logística
- **POST** `/api/churn/logreg`
  - **Body**: Mismo formato que `/api/churn/knn`
  - **Respuesta**:
  ```json
  {
    "model": "logistic_regression",
    "prediccion": 0,
    "probabilidad": 0.23,
    "mensaje": "El cliente tiene baja probabilidad de abandonar el servicio."
  }
  ```

### Credit Card - K-Means
- **POST** `/api/credit/kmeans`
  - **Body** (JSON):
  ```json
  {
    "Saldo": 1500.50,
    "Frecuencia_Saldo": 0.85,
    "Compras_Totales": 2000.0,
    "Compras_Contado": 500.0,
    "Compras_Cuotas": 1500.0,
    "Avances_Efectivo": 300.0,
    "Frecuencia_Compras": 0.6,
    "Frec_Compras_Contado": 0.2,
    "Frec_Compras_Cuotas": 0.4,
    "Frec_Avances": 0.1,
    "Transacciones_Avance": 5,
    "Transacciones_Compra": 25,
    "Limite_Credito": 5000.0,
    "Pagos_Realizados": 1200.75,
    "Pago_Minimo": 200.0,
    "Pct_Pago_Completo": 0.1
  }
  ```
  - **Respuesta**:
  ```json
  {
    "model": "kmeans",
    "cluster": 2,
    "segmentacion": {
      "cluster": 2,
      "nombre_segmento": "Cliente con dependencia de adelantos",
      "descripcion": "Cliente con balance medio, uso moderado y ligera dependencia de adelantos en efectivo.",
      "detalles": {
        "riesgo": "medio",
        "tipo_cliente": "rotativo",
        "recomendacion": "Ofrecer plan de consolidación de deuda."
      }
    }
  }
  ```

## ⚠️ Manejo de Errores

Todos los errores se devuelven en formato JSON:

```json
{
  "error": "VALIDATION_ERROR",
  "message": "El campo 'Saldo' es requerido y debe ser numérico."
}
```

Códigos de estado HTTP:
- `400`: Error de validación
- `404`: Ruta no encontrada
- `500`: Error interno del servidor

## 🔧 Tecnologías

- **Node.js**: Runtime de JavaScript
- **Express**: Framework web
- **TypeScript**: Superset tipado de JavaScript
- **Zod**: Validación de esquemas
- **CORS**: Middleware para Cross-Origin Resource Sharing
- **dotenv**: Manejo de variables de entorno
- **onnxruntime-node**: Runtime para ejecutar modelos ONNX

## 📝 Notas Importantes

- **Modelos ONNX requeridos**: Los modelos deben estar en la carpeta `models/` para que los endpoints funcionen
- **Formato de modelos**: Ver `MODELOS_ONNX.md` para detalles sobre el formato esperado de los modelos
- **Preprocesamiento**: Los datos se preprocesan automáticamente antes de pasarlos a los modelos
- **CORS**: Configurado para permitir peticiones desde `http://localhost:5173` (React por defecto)

## 🧪 Pruebas

### Con curl

**Churn KNN**:
```bash
curl -X POST http://localhost:3000/api/churn/knn \
  -H "Content-Type: application/json" \
  -d '{
    "adulto_mayor": 0,
    "meses_como_cliente": 12,
    "cargo_mensual": 75.35,
    "tiene_pareja": "Yes",
    "dependientes": "No",
    "tipo_internet": "Fiber optic",
    "seguridad_en_linea": "No",
    "respaldo_en_linea": "No",
    "proteccion_dispositivo": "No",
    "soporte_tecnico": "No",
    "tipo_contrato": "Month-to-month",
    "facturacion_electronica": "Yes",
    "metodo_pago": "Electronic check"
  }'
```

**Credit K-Means**:
```bash
curl -X POST http://localhost:3000/api/credit/kmeans \
  -H "Content-Type: application/json" \
  -d '{
    "Saldo": 1500.50,
    "Frecuencia_Saldo": 0.85,
    "Compras_Totales": 2000.0,
    "Compras_Contado": 500.0,
    "Compras_Cuotas": 1500.0,
    "Avances_Efectivo": 300.0,
    "Frecuencia_Compras": 0.6,
    "Frec_Compras_Contado": 0.2,
    "Frec_Compras_Cuotas": 0.4,
    "Frec_Avances": 0.1,
    "Transacciones_Avance": 5,
    "Transacciones_Compra": 25,
    "Limite_Credito": 5000.0,
    "Pagos_Realizados": 1200.75,
    "Pago_Minimo": 200.0,
    "Pct_Pago_Completo": 0.1
  }'
```

### Otras herramientas

También puedes probar con:
- **Postman**
- **Thunder Client** (VS Code)
- **Frontend React** (desde `http://localhost:5173`)

## 📚 Documentación Adicional

- **`MODELOS_ONNX.md`**: Guía detallada sobre cómo preparar y usar tus modelos ONNX
- **`package.json`**: Scripts disponibles y dependencias

## 🐛 Solución de Problemas

### El servidor no inicia
- Verifica que todas las dependencias estén instaladas: `npm install`
- Verifica que el puerto 3000 no esté en uso
- Revisa los logs en la consola para ver errores específicos

### Los modelos no se cargan
- Verifica que los archivos `.onnx` estén en la carpeta `models/`
- Verifica las rutas en `.env` si las personalizaste
- Revisa la consola del servidor para mensajes de error

### Errores de validación
- Verifica que todos los campos requeridos estén presentes
- Verifica que los tipos de datos sean correctos (números, strings, etc.)
- Revisa el formato esperado en la sección de Endpoints

Para más detalles sobre problemas con modelos ONNX, consulta `MODELOS_ONNX.md`.

