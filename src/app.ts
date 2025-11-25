import express, { Application } from 'express';
import cors from 'cors';
import { config } from './config/env';
import churnRoutes from './routes/churn.routes';
import creditRoutes from './routes/credit.routes';
import { errorHandler } from './middlewares/errorHandler';
import { notFoundHandler } from './middlewares/notFoundHandler';

/**
 * Configuración de la aplicación Express
 * Incluye middlewares, rutas y manejo de errores
 */
export const createApp = (): Application => {
  const app = express();

  // Middleware para parsear JSON
  app.use(express.json());

  // Configuración de CORS
  // Permite peticiones desde el frontend en React (puerto 5173 por defecto)
  app.use(
    cors({
      origin: config.corsOrigin,
      credentials: true,
    })
  );

  // Endpoint de salud (health check)
  app.get('/health', (req, res) => {
    res.status(200).json({ status: 'ok' });
  });

  // Rutas de la API
  app.use('/api/churn', churnRoutes);
  app.use('/api/credit', creditRoutes);

  // Middleware para rutas no encontradas (404)
  app.use(notFoundHandler);

  // Middleware de manejo de errores (debe ir al final)
  app.use(errorHandler);

  return app;
};

