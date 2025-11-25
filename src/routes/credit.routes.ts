import { Router } from 'express';
import { z } from 'zod';
import { predictCreditKmeans } from '../controllers/credit.controller';
import { validateRequest } from '../middlewares/validateRequest';

const router = Router();

/**
 * Esquema de validación para el payload de tarjeta de crédito usando Zod
 */
const creditCardRequestSchema = z.object({
  BALANCE: z.number().min(0),
  PURCHASES_FREQUENCY: z.number().min(0).max(1),
  CASH_ADVANCE: z.number().min(0),
  PAYMENTS: z.number().min(0),
  MINIMUM_PAYMENTS: z.number().min(0),
  PRC_FULL_PAYMENT: z.number().min(0).max(1),
  CREDIT_LIMIT: z.number().min(0),
});

/**
 * Ruta: POST /api/credit/kmeans
 * Descripción: Segmenta cliente de tarjeta de crédito usando K-Means simulado
 */
router.post(
  '/kmeans',
  validateRequest(creditCardRequestSchema),
  predictCreditKmeans
);

export default router;

