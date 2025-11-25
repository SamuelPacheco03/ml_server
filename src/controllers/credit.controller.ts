import { Request, Response, NextFunction } from 'express';
import { CreditKmeansService } from '../services/creditKmeans.service';
import { CreditCardRequest } from '../models/types';

// Instancia del servicio
const creditKmeansService = new CreditKmeansService();

/**
 * Controlador para el endpoint de segmentación de clientes usando K-Means
 * POST /api/credit/kmeans
 */
export const predictCreditKmeans = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const requestData: CreditCardRequest = req.body;
    const prediction = await creditKmeansService.predict(requestData);

    res.status(200).json(prediction);
  } catch (error) {
    next(error);
  }
};

