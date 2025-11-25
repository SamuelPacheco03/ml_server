import * as ort from 'onnxruntime-node';
import * as path from 'path';
import * as fs from 'fs';

/**
 * Utilidad para cargar modelos ONNX
 * Maneja la carga y cacheo de modelos para mejorar el rendimiento
 */
export class OnnxModelLoader {
  private static modelCache: Map<string, ort.InferenceSession> = new Map();

  /**
   * Carga un modelo ONNX desde un archivo
   * @param modelPath - Ruta al archivo .onnx
   * @param useCache - Si es true, usa cache para evitar cargar el modelo múltiples veces
   * @returns Sesión de inferencia del modelo ONNX
   */
  static async loadModel(
    modelPath: string,
    useCache: boolean = true
  ): Promise<ort.InferenceSession> {
    // Verificar si el modelo está en cache
    if (useCache && this.modelCache.has(modelPath)) {
      return this.modelCache.get(modelPath)!;
    }

    // Resolver la ruta absoluta
    const absolutePath = path.isAbsolute(modelPath)
      ? modelPath
      : path.join(process.cwd(), modelPath);

    // Verificar que el archivo existe
    if (!fs.existsSync(absolutePath)) {
      throw new Error(`Modelo ONNX no encontrado en: ${absolutePath}`);
    }

    try {
      // Cargar el modelo
      const session = await ort.InferenceSession.create(absolutePath, {
        executionProviders: ['cpu'], // Usar CPU (puedes cambiar a 'cuda' si tienes GPU)
      });

      // Guardar en cache si está habilitado
      if (useCache) {
        this.modelCache.set(modelPath, session);
      }

      console.log(`✅ Modelo ONNX cargado exitosamente: ${absolutePath}`);
      return session;
    } catch (error) {
      console.error(`❌ Error al cargar modelo ONNX: ${absolutePath}`, error);
      throw error;
    }
  }

  /**
   * Limpia el cache de modelos
   */
  static clearCache(): void {
    this.modelCache.clear();
  }

  /**
   * Verifica si un modelo está cargado en cache
   */
  static isModelCached(modelPath: string): boolean {
    return this.modelCache.has(modelPath);
  }
}

