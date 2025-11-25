import { Request, Response, NextFunction } from 'express';
import { ErrorResponse } from '../models/types';

/**
 * Middleware centralizado para manejo de errores
 * Captura todos los errores no manejados y devuelve una respuesta JSON consistente
 */
export const errorHandler = (
  err: Error | any,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  // Si la respuesta ya fue enviada, delegar al manejador de errores por defecto de Express
  if (res.headersSent) {
    return next(err);
  }

  // Determinar el código de estado HTTP
  const statusCode = err.statusCode || err.status || 500;
  
  // Determinar el tipo de error
  const errorType = err.errorType || 'INTERNAL_SERVER_ERROR';
  
  // Mensaje de error
  const message = err.message || 'Ha ocurrido un error interno en el servidor';

  // Log del error en desarrollo
  if (process.env.NODE_ENV === 'development') {
    console.error('Error:', err);
  }

  // Respuesta de error estandarizada
  const errorResponse: ErrorResponse = {
    error: errorType,
    message: message,
  };

  res.status(statusCode).json(errorResponse);
};

