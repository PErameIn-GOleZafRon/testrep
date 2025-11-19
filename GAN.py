import tensorflow as tf
import numpy as np
import os

# Загружаем датасеты
from Learn import data as learn_data
from Valid import data as valid_data

class ConditionalDigitGAN:
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        self.g_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    def build_generator(self):
        noise_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        label_input = tf.keras.layers.Input(shape=(10,))
        
        x = tf.keras.layers.concatenate([noise_input, label_input])
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(100, activation='sigmoid')(x)
        
        return tf.keras.Model([noise_input, label_input], x)
    
    def build_discriminator(self):
        image_input = tf.keras.layers.Input(shape=(100,))
        label_input = tf.keras.layers.Input(shape=(10,))
        
        x = tf.keras.layers.concatenate([image_input, label_input])
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        return tf.keras.Model([image_input, label_input], x)
    
    def generate_noise(self, batch_size):
        return tf.random.normal([batch_size, self.latent_dim])
    
    def train_step(self, real_images, real_labels):
        batch_size = tf.shape(real_images)[0]
        
        # Обучаем дискриминатор
        with tf.GradientTape() as d_tape:
            # Реальные изображения
            real_output = self.discriminator([real_images, real_labels], training=True)
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            
            # Сгенерированные изображения
            noise = self.generate_noise(batch_size)
            fake_images = self.generator([noise, real_labels], training=True)
            fake_output = self.discriminator([fake_images, real_labels], training=True)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            
            d_loss = (real_loss + fake_loss) / 2
            
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Обучаем генератор
        with tf.GradientTape() as g_tape:
            noise = self.generate_noise(batch_size)
            fake_images = self.generator([noise, real_labels], training=True)
            fake_output = self.discriminator([fake_images, real_labels], training=True)
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        return d_loss, g_loss
    
    def generate_digit(self, digit_class, num_samples=1):
        noise = self.generate_noise(num_samples)
        labels = tf.one_hot([digit_class] * num_samples, 10)
        generated = self.generator([noise, labels], training=False)
        # Применяем порог для четкости
        generated_binary = tf.cast(generated > 0.5, tf.float32)
        return generated_binary.numpy()[0]
    
    def save_weights(self, filename='Weights.py'):
        # Сохраняем веса генератора
        weights_dict = {}
        for i, layer in enumerate(self.generator.layers):
            if layer.weights:
                weights_dict[f'generator_{i}'] = [w.numpy().tolist() for w in layer.weights]
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('weights = {\n')
            for key, weights_list in weights_dict.items():
                f.write(f"    '{key}': {weights_list},\n")
            f.write('}\n')
        
        print(f"Веса сохранены в {filename}")
    
    def load_weights(self, filename='Weights.py'):
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден. Нужно сначала обучить модель.")
            return False
        
        # Импортируем веса
        from Weights import weights
        
        # Загружаем веса в генератор
        for i, layer in enumerate(self.generator.layers):
            if f'generator_{i}' in weights and layer.weights:
                layer_weights = weights[f'generator_{i}']
                layer.set_weights([np.array(w) for w in layer_weights])
        
        print(f"Веса загружены из {filename}")
        return True

def prepare_dataset():
    """Подготавливает данные из Learn.py и Valid.py"""
    X_train = []
    y_train = []
    
    # Обучающие данные
    for sample in learn_data:
        X_train.append(sample[:100])
        digit = np.argmax(sample[100:])
        y_train.append(digit)
    
    # Валидационные данные
    for sample in valid_data:
        X_train.append(sample[:100])
        digit = np.argmax(sample[100:])
        y_train.append(digit)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    
    # Преобразуем в one-hot encoding
    y_train_onehot = tf.one_hot(y_train, 10).numpy()
    
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_onehot))
    dataset = dataset.batch(32).shuffle(1000)
    
    print(f"Загружено {len(X_train)} примеров для обучения")
    return dataset

def format_output(digit_array, digit_class):
    """Форматирует вывод в нужном формате"""
    output = [0.0] * 10
    output[digit_class] = 1.0
    
    result = f"#{digit_class}\n[\n"
    
    # Пиксели в формате 10x10
    for row in range(10):
        start_idx = row * 10
        end_idx = start_idx + 10
        row_data = digit_array[start_idx:end_idx]
        result += "    " + ", ".join(map(str, row_data.astype(int))) + ",  "
        result += f"# {start_idx+1}-{end_idx}\n"
    
    # Выходные значения
    result += "    # Выходные значения (SoftMax)\n"
    result += "    " + ", ".join(map(str, output)) + "\n"
    result += "],\n"
    
    return result

def train_model():
    """Режим обучения"""
    print("Режим обучения...")
    
    # Подготовка данных
    dataset = prepare_dataset()
    
    # Создание и обучение модели
    gan = ConditionalDigitGAN()
    
    print("Начинаем обучение...")
    epochs = 2000
    
    for epoch in range(epochs):
        total_d_loss = 0
        total_g_loss = 0
        num_batches = 0
        
        for batch, (images, labels) in enumerate(dataset):
            d_loss, g_loss = gan.train_step(images, labels)
            total_d_loss += d_loss
            total_g_loss += g_loss
            num_batches += 1
        
        if epoch % 200 == 0:
            print(f"Эпоха {epoch}, D_loss: {total_d_loss/num_batches:.4f}, G_loss: {total_g_loss/num_batches:.4f}")
    
    # Сохраняем веса
    gan.save_weights()
    print("Обучение завершено!")
    
    return gan

def use_model():
    """Режим использования"""
    print("Режим использования...")
    
    # Создаем модель
    gan = ConditionalDigitGAN()
    
    # Пытаемся загрузить веса
    if not gan.load_weights():
        return
    
    while True:
        try:
            digit_class = int(input("Какую цифру сгенерировать? (0-9, -1 для выхода): "))
            if digit_class == -1:
                break
            if 0 <= digit_class <= 9:
                # Генерируем цифру
                generated_digit = gan.generate_digit(digit_class)
                
                # Выводим в нужном формате
                print("\n" + "="*50)
                print(format_output(generated_digit, digit_class))
                print("="*50 + "\n")
                
                # Показываем визуализацию
                show_digit_visualization(generated_digit, digit_class)
            else:
                print("Пожалуйста, введите число от 0 до 9")
        except ValueError:
            print("Пожалуйста, введите корректное число")

def show_digit_visualization(digit_array, digit_class):
    """Показывает визуализацию цифры"""
    try:
        import matplotlib.pyplot as plt
        
        digit_image = digit_array.reshape(10, 10)
        
        plt.figure(figsize=(4, 4))
        plt.imshow(digit_image, cmap='binary', vmin=0, vmax=1)
        plt.title(f'Сгенерированная цифра: {digit_class}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("(Для визуализации установите matplotlib: pip install matplotlib)")

def main():
    """Главное меню"""
    print("=" * 60)
    print("          ГЕНЕРАТОР ЦИФР НА ОСНОВЕ GAN")
    print("=" * 60)
    
    while True:
        print("\nМеню:")
        print("1 - Использовать нейросеть (генерация цифр)")
        print("2 - Обучить нейросеть")
        print("3 - Выход")
        
        choice = input("Выберите действие (1-3): ").strip()
        
        if choice == '1':
            use_model()
        elif choice == '2':
            gan = train_model()
            
            # Предлагаем сразу протестировать
            test = input("Хотите протестировать генерацию? (y/n): ").lower()
            if test == 'y':
                use_model()
                
        elif choice == '3':
            print("Выход из программы...")
            break
        else:
            print("Неверный выбор. Пожалуйста, введите 1, 2 или 3.")

if __name__ == "__main__":
    main()