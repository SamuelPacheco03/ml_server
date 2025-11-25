import { Request, Response, NextFunction } from 'express';
import { ErrorResponse } from '../models/types';

/**
 * Middleware para manejar rutas no encontradas (404)
 * Se ejecuta cuando ninguna ruta coincide con la petición
 */
export const notFoundHandler = (
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  const errorResponse: ErrorResponse = {
    error: 'NOT_FOUND',
    message: `La ruta ${req.method} ${req.originalUrl} no existe en este servidor.`,
  };

  res.status(404).json(errorResponse);
};

