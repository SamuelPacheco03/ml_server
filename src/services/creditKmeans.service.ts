// services/CreditKmeansService.ts

import {
  CreditCardRequest,
  KMeansResponse,
  ClusterDetails,
  SegmentacionCliente,
} from '../models/types';
import { OnnxModelLoader } from '../utils/onnxLoader';
import { preprocessCreditData, createTensor } from '../utils/preprocessing';
import { config } from '../config/env';
import * as ort from 'onnxruntime-node';
import * as fs from 'fs';
import * as path from 'path';

export class CreditKmeansService {
  private modelSession: ort.InferenceSession | null = null;
  private modelPath: string;
  private useOnnx: boolean = false;

  private clusterDescriptions: Map<
    number,
    {
      nombre_segmento: string;
      descripcion: string;
      detalles: ClusterDetails;
    }
  > = new Map([
    [
      0,
      {
        nombre_segmento: 'Usuarios en Riesgo por Avances',
        descripcion:
          'Clientes que usan la tarjeta principalmente como fuente de efectivo: muchos avances, alto saldo y pagos mínimos elevados. Muestran señales claras de estrés financiero.',
        detalles: {
          riesgo: 'alto',
          tipo_cliente: 'dependiente_avances',
          recomendacion:
            'Implementar campañas de educación financiera, ofrecer refinanciación o consolidación de deuda, reducir el acceso a avances y activar alertas tempranas.',
        },
      },
    ],
    [
      1,
      {
        nombre_segmento: 'Compradores Moderados y Disciplinados',
        descripcion:
          'Clientes ordenados, con compras totales moderadas, alta frecuencia de compras y casi sin avances. Mantienen saldos bajos y pagan de forma responsable.',
        detalles: {
          riesgo: 'bajo',
          tipo_cliente: 'moderado_responsable',
          recomendacion:
            'Ofrecer aumentos moderados de límite, programas de fidelización/cashback y productos premium de entrada.',
        },
      },
    ],
    [
      2,
      {
        nombre_segmento: 'Grandes Compradores (High Spenders Premium)',
        descripcion:
          'Clientes VIP con las compras más altas del portafolio, tanto al contado como en cuotas. Tienen límites elevados, pagan montos altos y prácticamente no usan avances.',
        detalles: {
          riesgo: 'bajo',
          tipo_cliente: 'premium_vip',
          recomendacion:
            'Ofrecer tarjetas gold/platinum, beneficios exclusivos (viajes, seguros, lounges), aumentos proactivos de límite y programas de lealtad premium.',
        },
      },
    ],
    [
      3,
      {
        nombre_segmento: 'Compradores Frecuentes de Ticket Medio',
        descripcion:
          'Clientes que usan mucho la tarjeta con frecuencia alta de compras y montos de ticket medio, principalmente al contado. Son muy rentables y casi no usan avances.',
        detalles: {
          riesgo: 'bajo',
          tipo_cliente: 'frecuente_rentable',
          recomendacion:
            'Ofrecer promociones por transacción, financiamiento a cuotas y mejores beneficios de cashback para reforzar su uso.',
        },
      },
    ],
    [
      4,
      {
        nombre_segmento: 'Clientes Inactivos o de Bajo Uso',
        descripcion:
          'Clientes con compras totales muy bajas, poca frecuencia de uso y sin avances. Pueden ser nuevos o poco interesados en la tarjeta.',
        detalles: {
          riesgo: 'bajo',
          tipo_cliente: 'inactivo_bajo_uso',
          recomendacion:
            'Lanzar campañas de activación, beneficios por primeras compras y promociones de 0% interés para incentivar el uso.',
        },
      },
    ],
  ]);


  constructor() {
    this.modelPath = config.models.creditKmeans;
    this.initializeModel();
  }

  private async initializeModel(): Promise<void> {
    try {
      const absolutePath = path.isAbsolute(this.modelPath)
        ? this.modelPath
        : path.join(process.cwd(), this.modelPath);

      if (fs.existsSync(absolutePath)) {
        this.modelSession = await OnnxModelLoader.loadModel(absolutePath);
        this.useOnnx = true;
        console.log('Usando modelo ONNX para Credit K-Means');
      } else {
        console.warn(
          `Modelo ONNX no encontrado en ${absolutePath}. Usando simulación.`
        );
        this.useOnnx = false;
      }
    } catch (err) {
      console.error('Error inicializando modelo ONNX:', err);
      this.useOnnx = false;
    }
  }

  async predict(request: CreditCardRequest): Promise<KMeansResponse> {
    if (this.useOnnx && this.modelSession) {
      try {
        return await this.predictWithOnnx(request);
      } catch (err) {
        console.error('Error en predicción ONNX:', err);
        if (config.useFallback) return this.predictSimulated(request);
        throw new Error('Error en predicción ONNX');
      }
    }

    if (config.useFallback) return this.predictSimulated(request);
    throw new Error('Modelo ONNX no disponible y fallback deshabilitado');
  }

  // Helper para construir objeto de segmentación
  private buildSegmentacion(cluster: number): SegmentacionCliente {
    const info = this.clusterDescriptions.get(cluster);

    if (!info) {
      return {
        cluster,
        nombre_segmento: `Segmento ${cluster}`,
        descripcion: `Cliente asignado al cluster ${cluster}.`,
        detalles: {
          riesgo: 'medio',
          tipo_cliente: 'estándar',
          recomendacion: 'Monitoreo regular y ofertas personalizadas.',
        },
      };
    }

    return {
      cluster,
      nombre_segmento: info.nombre_segmento,
      descripcion: info.descripcion,
      detalles: info.detalles,
    };
  }

  private async predictWithOnnx(
    request: CreditCardRequest
  ): Promise<KMeansResponse> {
    if (!this.modelSession) throw new Error('Modelo ONNX no inicializado');
    console.log('Prediciendo con modelo ONNX...');
    const features = preprocessCreditData(request);
    const inputTensor = createTensor(features, [1, features.length]);
    const inputName = this.modelSession.inputNames[0];

    const results = await this.modelSession.run({ [inputName]: inputTensor });
    const output = results[this.modelSession.outputNames[0]].data as any;

    const raw = output[0];
    const cluster =
      typeof raw === 'bigint' ? Number(raw) : Math.round(Number(raw));

    const segmentacion = this.buildSegmentacion(cluster);

    return {
      model: 'kmeans',
      cluster,
      segmentacion,
    };
  }

  private predictSimulated(request: CreditCardRequest): KMeansResponse {
    const {
      Saldo,
      Frecuencia_Compras,
      Avances_Efectivo,
      Pagos_Realizados,
      Pago_Minimo,
      Pct_Pago_Completo,
      Limite_Credito,
    } = request;

    const balanceRatio = Saldo / (Limite_Credito || 1);
    const paymentRatio = Pagos_Realizados / (Saldo || 1);
    const cashAdvanceRatio = Avances_Efectivo / (Saldo || 1);

    let cluster = 1;

    if (
      balanceRatio < 0.3 &&
      Frecuencia_Compras > 0.6 &&
      Pct_Pago_Completo > 0.5 &&
      cashAdvanceRatio < 0.1
    ) {
      // Premium
      cluster = 0;
    } else if (
      balanceRatio < 0.7 &&
      paymentRatio > 0.5 &&
      Frecuencia_Compras > 0.3 &&
      cashAdvanceRatio < 0.3
    ) {
      // Rotativo estable
      cluster = 1;
    } else if (
      cashAdvanceRatio > 0.4 ||
      (balanceRatio > 0.6 && Avances_Efectivo > Saldo * 0.3)
    ) {
      // Dependencia de adelantos
      cluster = 2;
    } else if (
      balanceRatio > 0.7 &&
      paymentRatio < 0.3 &&
      Pct_Pago_Completo < 0.2 &&
      Frecuencia_Compras < 0.3
    ) {
      // Alto riesgo
      cluster = 3;
    } else if (Frecuencia_Compras < 0.2 && balanceRatio < 0.2) {
      // Inactivo
      cluster = 4;
    }

    const segmentacion = this.buildSegmentacion(cluster);

    return {
      model: 'kmeans',
      cluster,
      segmentacion,
    };
  }
}
