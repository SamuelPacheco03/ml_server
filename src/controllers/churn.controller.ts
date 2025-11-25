import { Request, Response, NextFunction } from 'express';
import { ChurnKnnService } from '../services/churnKnn.service';
import { ChurnLogRegService } from '../services/churnLogReg.service';
import { ChurnRequest } from '../models/types';

// Instancias de los servicios
const churnKnnService = new ChurnKnnService();
const churnLogRegService = new ChurnLogRegService();

/**
 * Controlador para el endpoint de predicción de churn usando KNN
 * POST /api/churn/knn
 */
export const predictChurnKnn = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const requestData: ChurnRequest = req.body;

    const prediction = await churnKnnService.predict(requestData);

    res.status(200).json(prediction);
  } catch (error) {
    next(error);
  }
};

/**
 * Controlador para el endpoint de predicción de churn usando Regresión Logística
 * POST /api/churn/logreg
 */
export const predictChurnLogReg = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const requestData: ChurnRequest = req.body;

    const prediction = await churnLogRegService.predict(requestData);

    res.status(200).json(prediction);
  } catch (error) {
    next(error);
  }
};

