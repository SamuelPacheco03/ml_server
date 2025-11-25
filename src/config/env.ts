import dotenv from 'dotenv';

// Cargar variables de entorno desde .env
dotenv.config();

/**
 * Configuración de variables de entorno
 * Centraliza el acceso a las variables de entorno con valores por defecto
 */
export const config = {
  port: process.env.PORT || 3000,
  nodeEnv: process.env.NODE_ENV || 'development',
  corsOrigin: process.env.CORS_ORIGIN || 'http://localhost:5173',
  // Rutas a los modelos ONNX (opcionales, si no existen se usa simulación)
  models: {
    churnKnn: process.env.CHURN_KNN_MODEL_PATH || 'models/churn_knn.onnx',
    churnLogReg: process.env.CHURN_LOGREG_MODEL_PATH || 'models/churn_logreg.onnx',
    creditKmeans: process.env.CREDIT_KMEANS_MODEL_PATH || 'models/credit_kmeans.onnx',
  },
  // Si es true, usa simulación si el modelo no está disponible
  useFallback: process.env.USE_FALLBACK !== 'false',
};

