import { ChurnRequest, ChurnResponse } from '../models/types';
import { OnnxModelLoader } from '../utils/onnxLoader';
import { preprocessChurnData, createTensor } from '../utils/preprocessing';
import { config } from '../config/env';
import * as ort from 'onnxruntime-node';
import * as fs from 'fs';
import * as path from 'path';

export class ChurnLogRegService {
  private modelSession: ort.InferenceSession | null = null;
  private modelPath: string;
  private useOnnx = false;

  constructor() {
    this.modelPath = config.models.churnLogReg;
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
        console.log('✅ Usando modelo ONNX (Regresión Logística)');
      } else {
        console.warn(`⚠️ Modelo ONNX LogReg no encontrado en ${absolutePath}. Usando simulación.`);
      }
    } catch (error) {
      console.error('Error al cargar modelo ONNX LogReg:', error);
    }
  }

  async predict(request: ChurnRequest): Promise<ChurnResponse> {
    if (this.useOnnx && this.modelSession) {
      try {
        return await this.predictWithOnnx(request);
      } catch (err) {
        console.error('❌ Error ONNX LogReg, usando fallback:', err);
      }
    }

    return this.predictSimulated(request);
  }

  private async predictWithOnnx(request: ChurnRequest): Promise<ChurnResponse> {
    if (!this.modelSession) throw new Error('Modelo ONNX LogReg no inicializado');

    const features = preprocessChurnData(request);
    const inputTensor = createTensor(features, [1, features.length]);
    const inputName = this.modelSession.inputNames[0];

    const results = await this.modelSession.run({ [inputName]: inputTensor });

    const [labelName, probName] = this.modelSession.outputNames;

    const rawLabel = (results[labelName].data as any)[0];
    const prediccion = Number(rawLabel) as 0 | 1;

    const probs = Array.from(results[probName].data as any).map(Number);
    const probNoAbandono = probs[0];
    const probAbandono = probs[1];

    let probabilidadClase = prediccion === 1 ? probAbandono : probNoAbandono;
    probabilidadClase = Math.min(1, Math.max(0, probabilidadClase));

    return {
      model: 'logistic_regression',
      prediccion,
      probabilidad: Math.round(probabilidadClase * 100) / 100,
      mensaje: this.generateMessage(prediccion, probabilidadClase),
    };
  }

  private predictSimulated(request: ChurnRequest): ChurnResponse {
    let logit = -2.5;

    logit += -0.02 * request.meses_como_cliente;
    logit += 0.01 * request.cargo_mensual;
    logit += 0.3 * request.adulto_mayor;

    if (request.tipo_contrato === 'Month-to-month') logit += 1.2;
    else if (request.tipo_contrato === 'One year') logit += 0.3;
    else logit -= 0.5;

    if (request.metodo_pago === 'Electronic check') logit += 0.8;
    else if (request.metodo_pago === 'Mailed check') logit += 0.4;
    else logit -= 0.2;

    if (request.tipo_internet === 'Fiber optic') logit += 0.4;
    else if (request.tipo_internet === 'DSL') logit += 0.1;
    else logit -= 0.3;

    if (request.seguridad_en_linea === 'Yes') logit -= 0.2;
    if (request.respaldo_en_linea === 'Yes') logit -= 0.15;
    if (request.proteccion_dispositivo === 'Yes') logit -= 0.15;
    if (request.soporte_tecnico === 'Yes') logit -= 0.25;

    if (request.facturacion_electronica === 'No') logit += 0.3;
    if (request.tiene_pareja === 'No') logit += 0.1;
    if (request.dependientes === 'Yes') logit -= 0.1;

    const pAbandono = 1 / (1 + Math.exp(-logit));
    const pAb = Math.min(0.99, Math.max(0.01, pAbandono));
    const prediccion: 0 | 1 = pAb >= 0.5 ? 1 : 0;

    const probabilidadClase = prediccion === 1 ? pAb : 1 - pAb;

    return {
      model: 'logistic_regression',
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
