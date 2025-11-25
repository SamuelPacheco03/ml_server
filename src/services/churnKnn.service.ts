import { ChurnRequest, ChurnResponse } from '../models/types';
import { OnnxModelLoader } from '../utils/onnxLoader';
import { preprocessChurnData, createTensor } from '../utils/preprocessing';
import { config } from '../config/env';
import * as ort from 'onnxruntime-node';
import * as fs from 'fs';
import * as path from 'path';

export class ChurnKnnService {
  private modelSession: ort.InferenceSession | null = null;
  private modelPath: string;
  private useOnnx = false;

  constructor() {
    this.modelPath = config.models.churnKnn;
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
        console.log('Usando modelo ONNX (KNN)');
      } else {
        console.warn(`Modelo ONNX KNN no encontrado en ${absolutePath}. Usando simulación.`);
        this.useOnnx = false;
      }
    } catch (error) {
      console.error('Error al cargar modelo ONNX para Churn KNN:', error);
      this.useOnnx = false;
    }
  }

  async predict(request: ChurnRequest): Promise<ChurnResponse> {
    if (this.useOnnx && this.modelSession) {
      try {
        return await this.predictWithOnnx(request);
      } catch (error) {
        console.error('Error en predicción ONNX KNN, usando fallback:', error);
        if (config.useFallback) {
          return this.predictSimulated(request);
        }
        throw error;
      }
    }

    if (config.useFallback) {
      return this.predictSimulated(request);
    }

    throw new Error('Modelo ONNX KNN no disponible y fallback deshabilitado');
  }

  private async predictWithOnnx(request: ChurnRequest): Promise<ChurnResponse> {
    if (!this.modelSession) {
      throw new Error('Modelo ONNX KNN no inicializado');
    }

    const features = preprocessChurnData(request);
    const inputTensor = createTensor(features, [1, features.length]);
    const inputName = this.modelSession.inputNames[0];

    const results = await this.modelSession.run({ [inputName]: inputTensor });

    const [labelName, probName] = this.modelSession.outputNames;
    const labelTensor = results[labelName];
    const probTensor = results[probName];

    const rawLabel = (labelTensor.data as any)[0];
    const prediccion = Number(rawLabel) as 0 | 1;

    const probs = Array.from(probTensor.data as any).map(Number);
    const probNoAbandono = probs[0];
    const probAbandono = probs[1];

    let probabilidadClase = prediccion === 1 ? probAbandono : probNoAbandono;
    probabilidadClase = Math.min(1, Math.max(0, probabilidadClase));

    return {
      model: 'knn',
      prediccion,
      probabilidad: Math.round(probabilidadClase * 100) / 100,
      mensaje: this.generateMessage(prediccion, probabilidadClase),
    };
  }

  private predictSimulated(request: ChurnRequest): ChurnResponse {
    let churnScore = 0;

    if (request.tipo_contrato === 'Month-to-month') {
      churnScore += 0.3;
    } else if (request.tipo_contrato === 'One year') {
      churnScore += 0.1;
    } else {
      churnScore += 0.05;
    }

    if (request.tipo_internet === 'Fiber optic') {
      churnScore += 0.15;
    }

    if (request.metodo_pago === 'Electronic check') {
      churnScore += 0.2;
    }

    let serviciosAdicionales = 0;
    if (request.seguridad_en_linea === 'Yes') serviciosAdicionales++;
    if (request.respaldo_en_linea === 'Yes') serviciosAdicionales++;
    if (request.proteccion_dispositivo === 'Yes') serviciosAdicionales++;
    if (request.soporte_tecnico === 'Yes') serviciosAdicionales++;

    churnScore += (4 - serviciosAdicionales) * 0.1;

    if (request.meses_como_cliente < 12) {
      churnScore += 0.2;
    } else if (request.meses_como_cliente < 24) {
      churnScore += 0.1;
    }

    if (request.cargo_mensual > 100) {
      churnScore += 0.15;
    } else if (request.cargo_mensual < 30) {
      churnScore += 0.1;
    }

    if (request.facturacion_electronica === 'No') {
      churnScore += 0.1;
    }

    const probAbandono = Math.min(0.99, Math.max(0.01, churnScore));
    const prediccion: 0 | 1 = probAbandono >= 0.5 ? 1 : 0;

    const probabilidadClase = prediccion === 1 ? probAbandono : 1 - probAbandono;

    return {
      model: 'knn',
      prediccion,
      probabilidad: Math.round(probabilidadClase * 100) / 100,
      mensaje: this.generateMessage(prediccion, probabilidadClase),
    };
  }

  private generateMessage(prediccion: 0 | 1, probabilidadClase: number): string {
    if (prediccion === 1) {
      if (probabilidadClase >= 0.7) {
        return 'El cliente tiene alta probabilidad de abandonar el servicio.';
      }
      return 'El cliente tiene probabilidad moderada de abandonar el servicio.';
    }

    const probAbandono = 1 - probabilidadClase;

    if (probAbandono <= 0.3) {
      return 'El cliente tiene baja probabilidad de abandonar el servicio.';
    }
    return 'El cliente tiene probabilidad moderada-baja de abandonar el servicio.';
  }
}
