import { Request, Response, NextFunction } from 'express';
import { z, ZodError } from 'zod';
import { ErrorResponse } from '../models/types';

/**
 * Middleware factory para validar el body de la petición usando Zod
 * @param schema - Esquema de validación de Zod
 */
export const validateRequest = (schema: z.ZodSchema) => {
  return (req: Request, res: Response, next: NextFunction): void => {
    try {
      // Validar el body contra el esquema
      schema.parse(req.body);
      next();
    } catch (error) {
      if (error instanceof ZodError) {
        // Extraer el primer error de validación
        const firstError = error.errors[0];
        const field = firstError.path.join('.');
        
        const errorResponse: ErrorResponse = {
          error: 'VALIDATION_ERROR',
          message: `El campo '${field}' ${firstError.message.toLowerCase()}`,
        };

        res.status(400).json(errorResponse);
      } else {
        // Error inesperado
        const errorResponse: ErrorResponse = {
          error: 'VALIDATION_ERROR',
          message: 'Error de validación en la petición.',
        };

        res.status(400).json(errorResponse);
      }
    }
  };
};

