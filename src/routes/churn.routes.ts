import { Router } from 'express';
import { z } from 'zod';
import { predictChurnKnn, predictChurnLogReg } from '../controllers/churn.controller';
import { validateRequest } from '../middlewares/validateRequest';

const router = Router();

/**
 * Esquema de validación para el payload de churn usando Zod
 */
const churnRequestSchema = z.object({
  adulto_mayor: z.number().int().min(0).max(1),
  meses_como_cliente: z.number().int().min(0),
  cargo_mensual: z.number().min(0),
  tiene_pareja: z.enum(['Yes', 'No']),
  dependientes: z.enum(['Yes', 'No']),
  tipo_internet: z.string().min(1),
  seguridad_en_linea: z.enum(['Yes', 'No', 'No internet service']),
  respaldo_en_linea: z.enum(['Yes', 'No', 'No internet service']),
  proteccion_dispositivo: z.enum(['Yes', 'No', 'No internet service']),
  soporte_tecnico: z.enum(['Yes', 'No', 'No internet service']),
  tipo_contrato: z.string().min(1),
  facturacion_electronica: z.enum(['Yes', 'No']),
  metodo_pago: z.string().min(1),
});

/**
 * Ruta: POST /api/churn/knn
 * Descripción: Predice churn usando modelo KNN simulado
 */
router.post(
  '/knn',
  validateRequest(churnRequestSchema),
  predictChurnKnn
);

/**
 * Ruta: POST /api/churn/logreg
 * Descripción: Predice churn usando modelo de Regresión Logística simulado
 */
router.post(
  '/logreg',
  validateRequest(churnRequestSchema),
  predictChurnLogReg
);

export default router;

