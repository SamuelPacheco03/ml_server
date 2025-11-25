import { Request, Response, NextFunction } from 'express';
import { ErrorResponse } from '../models/types';

export const errorHandler = (
  err: Error | any,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  if (res.headersSent) {
    return next(err);
  }

  const statusCode = err.statusCode || err.status || 500;
  const errorType = err.errorType || 'INTERNAL_SERVER_ERROR';
  const message = err.message || 'Ha ocurrido un error interno en el servidor';

  if (process.env.NODE_ENV === 'development') {
    console.error('Error:', err);
  }

  const errorResponse: ErrorResponse = {
    error: errorType,
    message: message,
  };

  res.status(statusCode).json(errorResponse);
};

