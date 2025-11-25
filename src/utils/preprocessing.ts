import { ChurnRequest, CreditCardRequest } from '../models/types';
import * as ort from 'onnxruntime-node';

/**
 * Preprocesa los datos de churn para el modelo ONNX
 * Aplica MinMaxScaler y one-hot encoding como en el entrenamiento
 * @param request - Datos del cliente
 * @returns Array numérico de 21 posiciones listo para el modelo ONNX
 */
export function preprocessChurnData(request: ChurnRequest): number[] {
  // Ahora son 21 features (sin columna duplicada)
  const features = Array(21).fill(0);

  // MinMaxScaler parámetros del modelo (desde Python)
  const SCALE = {
    adulto_mayor:   { scale: 1.0,        min: 0.0 },
    meses_cliente:  { scale: 0.01388889, min: 0.0 },
    cargo_mensual:  { scale: 0.01001502, min: -0.18828242 },
  };

  const scaleValue = (value: number, scale: number, min: number): number =>
    value * scale + min;

  // 0–2: numéricas escaladas
  features[0] = scaleValue(
    request.adulto_mayor,
    SCALE.adulto_mayor.scale,
    SCALE.adulto_mayor.min
  );
  features[1] = scaleValue(
    request.meses_como_cliente,
    SCALE.meses_cliente.scale,
    SCALE.meses_cliente.min
  );
  features[2] = scaleValue(
    request.cargo_mensual,
    SCALE.cargo_mensual.scale,
    SCALE.cargo_mensual.min
  );

  // ❌ ya no hay columna duplicada en features[3]

  // 3–20: dummies, mismo orden que df_model SIN duplicado

  // 3 - tiene_pareja_Yes
  features[3] = request.tiene_pareja === 'Yes' ? 1 : 0;

  // 4 - dependientes_Yes
  features[4] = request.dependientes === 'Yes' ? 1 : 0;

  // 5–6 tipo_internet
  features[5] = request.tipo_internet === 'Fiber optic' ? 1 : 0; // tipo_internet_Fiber optic
  features[6] = request.tipo_internet === 'No' ? 1 : 0;          // tipo_internet_No

  // 7–8 seguridad_en_linea
  features[7] = request.seguridad_en_linea === 'No internet service' ? 1 : 0;
  features[8] = request.seguridad_en_linea === 'Yes' ? 1 : 0;

  // 9–10 respaldo_en_linea
  features[9]  = request.respaldo_en_linea === 'No internet service' ? 1 : 0;
  features[10] = request.respaldo_en_linea === 'Yes' ? 1 : 0;

  // 11–12 proteccion_dispositivo
  features[11] = request.proteccion_dispositivo === 'No internet service' ? 1 : 0;
  features[12] = request.proteccion_dispositivo === 'Yes' ? 1 : 0;

  // 13–14 soporte_tecnico
  features[13] = request.soporte_tecnico === 'No internet service' ? 1 : 0;
  features[14] = request.soporte_tecnico === 'Yes' ? 1 : 0;

  // 15–16 tipo_contrato
  features[15] = request.tipo_contrato === 'One year' ? 1 : 0;
  features[16] = request.tipo_contrato === 'Two year' ? 1 : 0;

  // 17 facturacion_electronica_Yes
  features[17] = request.facturacion_electronica === 'Yes' ? 1 : 0;

  // 18–20 metodo_pago
  features[18] = request.metodo_pago === 'Credit card (automatic)' ? 1 : 0;
  features[19] = request.metodo_pago === 'Electronic check' ? 1 : 0;
  features[20] = request.metodo_pago === 'Mailed check' ? 1 : 0;

  return features;
}

/**
 * Preprocesa los datos de tarjeta de crédito para el modelo K-Means ONNX
 */
export function preprocessCreditData(request: CreditCardRequest): number[] {
  return [
    request.BALANCE,
    request.PURCHASES_FREQUENCY,
    request.CASH_ADVANCE,
    request.PAYMENTS,
    request.MINIMUM_PAYMENTS,
    request.PRC_FULL_PAYMENT,
    request.CREDIT_LIMIT,
  ];
}

/**
 * Crea un tensor de ONNX desde un array de números
 */
export function createTensor(data: number[], shape: number[]): ort.Tensor {
  const floatData = data.map(val => {
    const num = Number(val);
    if (isNaN(num) || !isFinite(num)) {
      return 0;
    }
    return num;
  });

  const float32Array = new Float32Array(floatData);
  return new ort.Tensor('float32', float32Array, shape);
}
