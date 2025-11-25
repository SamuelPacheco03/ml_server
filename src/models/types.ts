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
  prediccion: 0 | 1;
  probabilidad: number;
  mensaje: string;
}

/**
 * Payload para segmentación de clientes de tarjeta de crédito (K-Means)
 */
export interface CreditCardRequest {
  Saldo: number;
  Frecuencia_Saldo: number;
  Compras_Totales: number;
  Compras_Contado: number;
  Compras_Cuotas: number;
  Avances_Efectivo: number;
  Frecuencia_Compras: number;
  Frec_Compras_Contado: number;
  Frec_Compras_Cuotas: number;
  Frec_Avances: number;
  Transacciones_Avance: number;
  Transacciones_Compra: number;
  Limite_Credito: number;
  Pagos_Realizados: number;
  Pago_Minimo: number;
  Pct_Pago_Completo: number;
}

export type RiesgoNivel = 'bajo' | 'medio' | 'alto';

/**
 * Detalles de un cluster
 */
export interface ClusterDetails {
  riesgo: RiesgoNivel;
  tipo_cliente: string;
  recomendacion: string;
}

export interface SegmentacionCliente {
  cluster: number;
  nombre_segmento: string;
  descripcion: string;
  detalles: ClusterDetails;
}


/**
 * Respuesta de segmentación K-Means
 */
export interface KMeansResponse {
  model: 'kmeans';
  cluster: number;
  segmentacion: SegmentacionCliente;
  features_normalizadas?: Record<string, number>;
}

/**
 * Respuesta de error estándar
 */
export interface ErrorResponse {
  error: string;
  message: string;
}

