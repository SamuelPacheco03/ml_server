/**
 * Tipos e interfaces TypeScript para los modelos de ML
 */

/**
 * Payload para predicción de churn (abandono de servicio)
 * Usado tanto para KNN como para Regresión Logística
 */
export interface ChurnRequest {
  adulto_mayor: number;
  meses_como_cliente: number;
  cargo_mensual: number;
  tiene_pareja: 'Yes' | 'No';
  dependientes: 'Yes' | 'No';
  tipo_internet: string;
  seguridad_en_linea: 'Yes' | 'No' | 'No internet service';
  respaldo_en_linea: 'Yes' | 'No' | 'No internet service';
  proteccion_dispositivo: 'Yes' | 'No' | 'No internet service';
  soporte_tecnico: 'Yes' | 'No' | 'No internet service';
  tipo_contrato: string;
  facturacion_electronica: 'Yes' | 'No';
  metodo_pago: string;
}

/**
 * Respuesta de predicción de churn
 */
export interface ChurnResponse {
  model: 'knn' | 'logistic_regression';
  prediccion: 0 | 1; // 0 = no abandona, 1 = abandona
  probabilidad: number; // valor entre 0 y 1
  mensaje: string;
}

/**
 * Payload para segmentación de clientes de tarjeta de crédito (K-Means)
 */
export interface CreditCardRequest {
  BALANCE: number;
  PURCHASES_FREQUENCY: number;
  CASH_ADVANCE: number;
  PAYMENTS: number;
  MINIMUM_PAYMENTS: number;
  PRC_FULL_PAYMENT: number;
  CREDIT_LIMIT: number;
}

/**
 * Detalles de un cluster
 */
export interface ClusterDetails {
  riesgo: 'bajo' | 'medio' | 'alto';
  tipo_cliente: string;
  recomendacion: string;
}

/**
 * Respuesta de segmentación K-Means
 */
export interface KMeansResponse {
  model: 'kmeans';
  cluster: number;
  descripcion: string;
  detalles_cluster: ClusterDetails;
}

/**
 * Respuesta de error estándar
 */
export interface ErrorResponse {
  error: string;
  message: string;
}

