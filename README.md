# ML Server - API Backend para Modelos de Machine Learning

API backend completa construida con Node.js, Express y TypeScript que simula 3 modelos de machine learning para predicción y segmentación de clientes.

## 🚀 Características

- **Arquitectura en capas**: Routes, Controllers, Services
- **TypeScript**: Código tipado y seguro
- **Validación de datos**: Usando Zod para validar payloads
- **Manejo de errores centralizado**: Middlewares para errores y rutas no encontradas
- **CORS configurado**: Listo para consumir desde frontend React
- **Modelos ML con soporte ONNX**:
  - KNN para predicción de churn (soporta modelos ONNX reales)
  - Regresión Logística para predicción de churn (soporta modelos ONNX reales)
  - K-Means para segmentación de clientes de tarjeta de crédito (soporta modelos ONNX reales)
  - **Fallback automático**: Si no hay modelo ONNX, usa simulación

## 📁 Estructura del Proyecto

```
src/
├── app.ts                 # Configuración de Express
├── server.ts              # Arranque del servidor
├── config/
│   └── env.ts            # Variables de entorno
├── controllers/
│   ├── churn.controller.ts
│   └── credit.controller.ts
├── middlewares/
│   ├── errorHandler.ts
│   ├── notFoundHandler.ts
│   └── validateRequest.ts
├── models/
│   └── types.ts          # Interfaces TypeScript
├── routes/
│   ├── churn.routes.ts
│   └── credit.routes.ts
└── services/
    ├── churnKnn.service.ts
    ├── churnLogReg.service.ts
    └── creditKmeans.service.ts
```

## 🛠️ Instalación

1. Instalar dependencias:
```bash
npm install
```

2. Crear archivo `.env` (opcional, tiene valores por defecto):
```env
PORT=3000
NODE_ENV=development
CORS_ORIGIN=http://localhost:5173
```

3. Ejecutar en modo desarrollo:
```bash
npm run dev
```

4. Compilar TypeScript:
```bash
npm run build
```

5. Ejecutar versión compilada:
```bash
npm start
```

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
  - **Body**: Mismo formato que `/knn`
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
    "BALANCE": 1500.50,
    "PURCHASES_FREQUENCY": 0.45,
    "CASH_ADVANCE": 300.0,
    "PAYMENTS": 1200.75,
    "MINIMUM_PAYMENTS": 200.0,
    "PRC_FULL_PAYMENT": 0.1,
    "CREDIT_LIMIT": 5000.0
  }
  ```
  - **Respuesta**:
  ```json
  {
    "model": "kmeans",
    "cluster": 2,
    "descripcion": "Cliente con balance medio, uso moderado y ligera dependencia de adelantos.",
    "detalles_cluster": {
      "riesgo": "medio",
      "tipo_cliente": "rotativo",
      "recomendacion": "Ofrecer plan de consolidación de deuda."
    }
  }
  ```

## ⚠️ Manejo de Errores

Todos los errores se devuelven en formato JSON:

```json
{
  "error": "VALIDATION_ERROR",
  "message": "El campo 'BALANCE' es requerido y debe ser numérico."
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

## 📝 Notas

- **Soporte para modelos ONNX reales**: Coloca tus modelos `.onnx` en la carpeta `models/` y se cargarán automáticamente
- **Fallback automático**: Si no hay modelo ONNX disponible, usa simulación
- Ver `MODELOS_ONNX.md` para instrucciones detalladas sobre cómo usar tus modelos ONNX
- La lógica de predicción es determinista y consistente para facilitar pruebas

## 🧪 Pruebas

Puedes probar los endpoints usando herramientas como:
- **Postman**
- **curl**
- **Thunder Client** (VS Code)
- **Frontend React** (desde `http://localhost:5173`)

Ejemplo con curl:
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

