/**
 * Custom TF.js layer for FaceNet Inception-ResNet "ScaleSum" Lambda layers.
 * The Keras model uses Lambda(scale_sum) with two inputs: output = x + scale * y.
 * We register this as "Lambda" so loadLayersModel can deserialize the model.
 */
import * as tf from "@tensorflow/tfjs";

const className = "Lambda";

interface LambdaConfig {
  name?: string;
  trainable?: boolean;
  arguments?: { scale?: number };
  [key: string]: unknown;
}

export class ScaleSumLambdaLayer extends tf.layers.Layer {
  static readonly className = className;
  scale: number;

  constructor(config: LambdaConfig) {
    super(config as tf.serialization.ConfigDict);
    const args = (config.arguments || {}) as { scale?: number };
    this.scale = typeof args.scale === "number" ? args.scale : 0.17;
  }

  getClassName(): string {
    return className;
  }

  call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return tf.tidy(() => {
      const arr = Array.isArray(inputs) ? inputs : [inputs];
      const [a, b] = arr;
      if (!a || !b) {
        throw new Error("ScaleSumLambda expects two inputs");
      }
      return tf.add(a, tf.mul(this.scale, b));
    });
  }

  getConfig(): tf.serialization.ConfigDict {
    const config = super.getConfig();
    (config as LambdaConfig).arguments = { scale: this.scale };
    return config;
  }

  /** @nocollapse */
  static fromConfig<T extends tf.serialization.Serializable>(
    cls: tf.serialization.SerializableConstructor<T>,
    config: tf.serialization.ConfigDict
  ): T {
    return new (cls as unknown as new (config: LambdaConfig) => T)(
      config as LambdaConfig
    ) as T;
  }
}

/** Call once before loadLayersModel so the FaceNet model can load. */
export function registerScaleSumLambda(): void {
  tf.serialization.registerClass(ScaleSumLambdaLayer);
}
