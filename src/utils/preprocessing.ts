import { ChurnRequest, CreditCardRequest } from '../models/types';
import * as ort from 'onnxruntime-node';

/**
 * Preprocesa los datos de churn para el modelo ONNX
 * Aplica MinMaxScaler y one-hot encoding como en el entrenamiento
 */
export function preprocessChurnData(request: ChurnRequest): number[] {
  const features = Array(21).fill(0);

  const SCALE = {
    adulto_mayor: { scale: 1.0, min: 0.0 },
    meses_cliente: { scale: 0.01388889, min: 0.0 },
    cargo_mensual: { scale: 0.01001502, min: -0.18828242 },
  };

  const scaleValue = (value: number, scale: number, min: number): number =>
    value * scale + min;

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

  // 3 - tiene_pareja_Yes
  features[3] = request.tiene_pareja === 'Yes' ? 1 : 0;

  // 4 - dependientes_Yes
  features[4] = request.dependientes === 'Yes' ? 1 : 0;

  // 5–6 tipo_internet
  features[5] = request.tipo_internet === 'Fiber optic' ? 1 : 0;
  features[6] = request.tipo_internet === 'No' ? 1 : 0;

  // 7–8 seguridad_en_linea
  features[7] = request.seguridad_en_linea === 'No internet service' ? 1 : 0;
  features[8] = request.seguridad_en_linea === 'Yes' ? 1 : 0;

  // 9–10 respaldo_en_linea
  features[9] = request.respaldo_en_linea === 'No internet service' ? 1 : 0;
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

const CREDIT_SCALER = {
  mean: [
    1.56447483e+03, // Saldo
    8.77270726e-01, // Frecuencia_Saldo
    1.00320483e+03, // Compras_Totales
    5.92437371e+02, // Compras_Contado
    4.11067645e+02, // Compras_Cuotas
    9.78871112e+02, // Avances_Efectivo
    4.90350548e-01, // Frecuencia_Compras
    2.02457684e-01, // Frec_Compras_Contado
    3.64437342e-01, // Frec_Compras_Cuotas
    1.35144200e-01, // Frec_Avances
    3.24882682e+00, // Transacciones_Avance
    1.47098324e+01, // Transacciones_Compra
    4.49428247e+03, // Limite_Credito
    1.73314385e+03, // Pagos_Realizados
    8.44906767e+02, // Pago_Minimo
    1.53714648e-01  // Pct_Pago_Completo
  ],
  scale: [
    2.08141559e+03,
    2.36890767e-01,
    2.13651541e+03,
    1.65979518e+03,
    9.04287592e+02,
    2.09704671e+03,
    4.01348324e-01,
    2.98319398e-01,
    3.97425575e-01,
    2.00110208e-01,
    6.82426547e+00,
    2.48562604e+01,
    3.63844342e+03,
    2.89490202e+03,
    2.33266199e+03,
    2.92482855e-01
  ]
};

function scaleStandard(value: number, index: number): number {
  return (value - CREDIT_SCALER.mean[index]) / CREDIT_SCALER.scale[index];
}

export function preprocessCreditData(request: CreditCardRequest): number[] {
  return [
    scaleStandard(request.Saldo, 0),
    scaleStandard(request.Frecuencia_Saldo, 1),
    scaleStandard(request.Compras_Totales, 2),
    scaleStandard(request.Compras_Contado, 3),
    scaleStandard(request.Compras_Cuotas, 4),
    scaleStandard(request.Avances_Efectivo, 5),
    scaleStandard(request.Frecuencia_Compras, 6),
    scaleStandard(request.Frec_Compras_Contado, 7),
    scaleStandard(request.Frec_Compras_Cuotas, 8),
    scaleStandard(request.Frec_Avances, 9),
    scaleStandard(request.Transacciones_Avance, 10),
    scaleStandard(request.Transacciones_Compra, 11),
    scaleStandard(request.Limite_Credito, 12),
    scaleStandard(request.Pagos_Realizados, 13),
    scaleStandard(request.Pago_Minimo, 14),
    scaleStandard(request.Pct_Pago_Completo, 15)
  ];
}

/**
 * Crea un tensor de ONNX desde un array de números
 */
export function createTensor(data: number[], shape: number[]): ort.Tensor {
  const floatData = data.map(val => {
    const num = Number(val);
    if (isNaN(num) || !isFinite(num)) return 0;
    return num;
  });

  const float32Array = new Float32Array(floatData);
  return new ort.Tensor('float32', float32Array, shape);
}
