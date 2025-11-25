import { createApp } from './app';
import { config } from './config/env';

/**
 * Archivo principal para arrancar el servidor
 * Crea la aplicación Express y la pone a escuchar en el puerto configurado
 */
const startServer = (): void => {
  const app = createApp();
  const port = config.port;

  app.listen(port, () => {
    console.log(`Servidor corriendo en http://localhost:${port}`);
    console.log(`Endpoints disponibles:`);
    console.log(`   - GET  /health`);
    console.log(`   - POST /api/churn/knn`);
    console.log(`   - POST /api/churn/logreg`);
    console.log(`   - POST /api/credit/kmeans`);
    console.log(`CORS habilitado para: ${config.corsOrigin}`);
  });
};

// Arrancar el servidor
startServer();

