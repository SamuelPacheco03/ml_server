import { Router } from 'express';
import { z } from 'zod';
import { predictCreditKmeans } from '../controllers/credit.controller';
import { validateRequest } from '../middlewares/validateRequest';

const router = Router();

const creditCardRequestSchema = z.object({
  Saldo: z.number().min(0),
  Frecuencia_Saldo: z.number().min(0),
  Compras_Totales: z.number().min(0),
  Compras_Contado: z.number().min(0),
  Compras_Cuotas: z.number().min(0),
  Avances_Efectivo: z.number().min(0),
  Frecuencia_Compras: z.number().min(0),
  Frec_Compras_Contado: z.number().min(0),
  Frec_Compras_Cuotas: z.number().min(0),
  Frec_Avances: z.number().min(0),
  Transacciones_Avance: z.number().min(0),
  Transacciones_Compra: z.number().min(0),
  Limite_Credito: z.number().min(0),
  Pagos_Realizados: z.number().min(0),
  Pago_Minimo: z.number().min(0),
  Pct_Pago_Completo: z.number().min(0).max(1),
});

router.post(
  '/kmeans',
  validateRequest(creditCardRequestSchema),
  predictCreditKmeans
);

export default router;

