import { CreditCardRequest, KMeansResponse, ClusterDetails } from '../models/types';
import { OnnxModelLoader } from '../utils/onnxLoader';
import { preprocessCreditData, createTensor } from '../utils/preprocessing';
import { config } from '../config/env';
import * as ort from 'onnxruntime-node';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Servicio que usa un modelo K-Means ONNX para segmentación de clientes de tarjeta de crédito
 * Si el modelo ONNX no está disponible, usa simulación como fallback
 */
export class CreditKmeansService {
  private modelSession: ort.InferenceSession | null = null;
  private modelPath: string;
  private useOnnx: boolean = false;

  // Mapeo de clusters a descripciones (para cuando usamos ONNX)
  private clusterDescriptions: Map<number, { descripcion: string; detalles: ClusterDetails }> = new Map([
    [0, {
      descripcion: 'Cliente premium con excelente comportamiento de pago y uso responsable del crédito.',
      detalles: {
        riesgo: 'bajo',
        tipo_cliente: 'premium',
        recomendacion: 'Ofrecer aumento de límite y beneficios exclusivos.',
      }
    }],
    [1, {
      descripcion: 'Cliente rotativo estable con balance medio y pagos regulares.',
      detalles: {
        riesgo: 'medio',
        tipo_cliente: 'rotativo_estable',
        recomendacion: 'Mantener comunicación activa y ofrecer programas de lealtad.',
      }
    }],
    [2, {
      descripcion: 'Cliente con balance medio, uso moderado y ligera dependencia de adelantos.',
      detalles: {
        riesgo: 'medio',
        tipo_cliente: 'rotativo',
        recomendacion: 'Ofrecer plan de consolidación de deuda.',
      }
    }],
    [3, {
      descripcion: 'Cliente con balance alto, pagos mínimos y bajo uso del crédito.',
      detalles: {
        riesgo: 'alto',
        tipo_cliente: 'alto_riesgo',
        recomendacion: 'Implementar estrategia de recuperación y reestructuración de deuda.',
      }
    }],
    [4, {
      descripcion: 'Cliente inactivo o nuevo con bajo uso del crédito.',
      detalles: {
        riesgo: 'bajo',
        tipo_cliente: 'inactivo',
        recomendacion: 'Campaña de activación y ofertas promocionales.',
      }
    }],
  ]);

  constructor() {
    this.modelPath = config.models.creditKmeans;
    this.initializeModel();
  }

  /**
   * Inicializa el modelo ONNX si está disponible
   */
  private async initializeModel(): Promise<void> {
    try {
      const absolutePath = path.isAbsolute(this.modelPath)
        ? this.modelPath
        : path.join(process.cwd(), this.modelPath);

      if (fs.existsSync(absolutePath)) {
        this.modelSession = await OnnxModelLoader.loadModel(absolutePath);
        this.useOnnx = true;
        console.log('✅ Usando modelo ONNX para Credit K-Means');
      } else {
        console.warn(
          `⚠️  Modelo ONNX no encontrado en ${absolutePath}. Usando simulación.`
        );
        this.useOnnx = false;
      }
    } catch (error) {
      console.error('Error al cargar modelo ONNX para Credit K-Means:', error);
      this.useOnnx = false;
    }
  }

  /**
   * Asigna un cliente a un cluster usando modelo ONNX o simulación
   * @param request - Datos del cliente de tarjeta de crédito
   * @returns Cluster asignado con descripción y detalles
   */
  async predict(request: CreditCardRequest): Promise<KMeansResponse> {
    // Intentar usar modelo ONNX si está disponible
    if (this.useOnnx && this.modelSession) {
      try {
        return await this.predictWithOnnx(request);
      } catch (error) {
        console.error('Error en predicción ONNX, usando fallback:', error);
        if (config.useFallback) {
          return this.predictSimulated(request);
        }
        throw error;
      }
    }

    // Usar simulación como fallback
    if (config.useFallback) {
      return this.predictSimulated(request);
    }

    throw new Error('Modelo ONNX no disponible y fallback deshabilitado');
  }

  /**
   * Predicción usando modelo ONNX real
   */
  private async predictWithOnnx(request: CreditCardRequest): Promise<KMeansResponse> {
    if (!this.modelSession) {
      throw new Error('Modelo ONNX no inicializado');
    }

    // Preprocesar datos
    const features = preprocessCreditData(request);
    
    // Crear tensor de entrada
    const inputTensor = createTensor(features, [1, features.length]);

    // Obtener el nombre del input del modelo
    const inputName = this.modelSession.inputNames[0];

    // Ejecutar inferencia
    const feeds = { [inputName]: inputTensor };
    const results = await this.modelSession.run(feeds);

    // Obtener la salida del modelo (cluster asignado)
    const outputName = this.modelSession.outputNames[0];
    const output = results[outputName];

    // Extraer el cluster predicho
    const outputData = output.data as Float32Array;
    const cluster = Math.round(outputData[0]); // K-Means devuelve el índice del cluster

    // Obtener descripción del cluster
    const clusterInfo = this.clusterDescriptions.get(cluster) || {
      descripcion: `Cliente asignado al cluster ${cluster}.`,
      detalles: {
        riesgo: 'medio' as const,
        tipo_cliente: 'estándar',
        recomendacion: 'Monitoreo regular y ofertas personalizadas según comportamiento.',
      }
    };

    return {
      model: 'kmeans',
      cluster,
      descripcion: clusterInfo.descripcion,
      detalles_cluster: clusterInfo.detalles,
    };
  }

  /**
   * Predicción usando simulación (fallback)
   */
  private predictSimulated(request: CreditCardRequest): KMeansResponse {
    const {
      BALANCE,
      PURCHASES_FREQUENCY,
      CASH_ADVANCE,
      PAYMENTS,
      MINIMUM_PAYMENTS,
      PRC_FULL_PAYMENT,
      CREDIT_LIMIT,
    } = request;

    // Calcular ratios importantes
    const balanceRatio = BALANCE / CREDIT_LIMIT;
    const paymentRatio = PAYMENTS / (BALANCE || 1);
    const cashAdvanceRatio = CASH_ADVANCE / (BALANCE || 1);

    // Asignar cluster basado en reglas
    let cluster: number;
    let descripcion: string;
    let detalles_cluster: ClusterDetails;

    if (
      balanceRatio < 0.3 &&
      PURCHASES_FREQUENCY > 0.6 &&
      PRC_FULL_PAYMENT > 0.5 &&
      cashAdvanceRatio < 0.1
    ) {
      cluster = 0;
      descripcion = 'Cliente premium con excelente comportamiento de pago y uso responsable del crédito.';
      detalles_cluster = {
        riesgo: 'bajo',
        tipo_cliente: 'premium',
        recomendacion: 'Ofrecer aumento de límite y beneficios exclusivos.',
      };
    } else if (
      balanceRatio >= 0.3 &&
      balanceRatio < 0.7 &&
      paymentRatio > 0.5 &&
      PURCHASES_FREQUENCY > 0.3 &&
      cashAdvanceRatio < 0.3
    ) {
      cluster = 1;
      descripcion = 'Cliente rotativo estable con balance medio y pagos regulares.';
      detalles_cluster = {
        riesgo: 'medio',
        tipo_cliente: 'rotativo_estable',
        recomendacion: 'Mantener comunicación activa y ofrecer programas de lealtad.',
      };
    } else if (
      cashAdvanceRatio > 0.4 ||
      (balanceRatio > 0.6 && CASH_ADVANCE > BALANCE * 0.3)
    ) {
      cluster = 2;
      descripcion = 'Cliente con balance medio, uso moderado y ligera dependencia de adelantos.';
      detalles_cluster = {
        riesgo: 'medio',
        tipo_cliente: 'rotativo',
        recomendacion: 'Ofrecer plan de consolidación de deuda.',
      };
    } else if (
      balanceRatio > 0.7 &&
      paymentRatio < 0.3 &&
      PRC_FULL_PAYMENT < 0.2 &&
      PURCHASES_FREQUENCY < 0.3
    ) {
      cluster = 3;
      descripcion = 'Cliente con balance alto, pagos mínimos y bajo uso del crédito.';
      detalles_cluster = {
        riesgo: 'alto',
        tipo_cliente: 'alto_riesgo',
        recomendacion: 'Implementar estrategia de recuperación y reestructuración de deuda.',
      };
    } else if (PURCHASES_FREQUENCY < 0.2 && balanceRatio < 0.2) {
      cluster = 4;
      descripcion = 'Cliente inactivo o nuevo con bajo uso del crédito.';
      detalles_cluster = {
        riesgo: 'bajo',
        tipo_cliente: 'inactivo',
        recomendacion: 'Campaña de activación y ofertas promocionales.',
      };
    } else {
      cluster = 1;
      descripcion = 'Cliente estándar con comportamiento típico de uso de tarjeta de crédito.';
      detalles_cluster = {
        riesgo: 'medio',
        tipo_cliente: 'estándar',
        recomendacion: 'Monitoreo regular y ofertas personalizadas según comportamiento.',
      };
    }

    return {
      model: 'kmeans',
      cluster,
      descripcion,
      detalles_cluster,
    };
  }
}
