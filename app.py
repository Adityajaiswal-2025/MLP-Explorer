import os
import math
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from sklearn.datasets import make_moons, make_circles
from sklearn.decomposition import PCA

app = Flask(__name__)

# Suppress TensorFlow logging to keep terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def generate_xor(n_samples=200, noise=0.1):
    """
    Generates a 2D XOR dataset.
    """
    n_per_cluster = n_samples // 4
    centers = np.array([
        [0.7, 0.7], [-0.7, -0.7],  # Class 1 (Quadrant I and III)
        [0.7, -0.7], [-0.7, 0.7]   # Class 0 (Quadrant II and IV)
    ])
    X_list = []
    y_list = []
    
    # Seed for reproducibility
    np.random.seed(42)
    
    for idx, center in enumerate(centers):
        c_noise = np.random.normal(0, noise, (n_per_cluster, 2))
        X_list.append(center + c_noise)
        # First two centers are Class 1, next two are Class 0
        y_list.append(np.ones(n_per_cluster) if idx < 2 else np.zeros(n_per_cluster))
        
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y

def generate_spiral(n_samples=200, noise=0.1):
    """
    Generates a 2D interleaving spiral dataset.
    """
    n_per_class = n_samples // 2
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples)
    
    # Seed for reproducibility
    np.random.seed(42)
    
    for c in range(2):
        # Angle goes from 0 to 2.5 * pi
        theta = np.linspace(0.0, 2.5 * np.pi, n_per_class)
        # Radius goes from 0.1 to 1.3
        r = (theta / (2.5 * np.pi)) * 1.3 + 0.1
        
        # Add random noise
        r_noise = np.random.normal(0, noise * 0.1, n_per_class)
        theta_noise = np.random.normal(0, noise * 0.1, n_per_class)
        
        X[c*n_per_class : (c+1)*n_per_class, 0] = (r + r_noise) * np.cos(theta + c*np.pi + theta_noise)
        X[c*n_per_class : (c+1)*n_per_class, 1] = (r + r_noise) * np.sin(theta + c*np.pi + theta_noise)
        y[c*n_per_class : (c+1)*n_per_class] = c
        
    return X, y

def generate_moons(n_samples=200, noise=0.1):
    """
    Generates a 2D interleaving moons dataset, centered and scaled.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    # Centering and scaling for better MLP visualization
    X = (X - np.array([0.5, 0.25])) * 1.3
    return X, y

def generate_circles(n_samples=200, noise=0.1):
    """
    Generates a 2D concentric circles dataset.
    """
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    # Scale for standard size
    X = X * 1.5
    return X, y

class HistoryRecorder(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to record weights, grid decisions,
    and hidden activations at regular intervals.
    """
    def __init__(self, X, y, grid_points):
        super().__init__()
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.grid_points = grid_points.astype(np.float32)
        self.records = []
        self.feature_extractor = None

    def on_train_begin(self, logs=None):
        # Extract Dense layers (which represent input->hidden and hidden->hidden, plus hidden->output)
        dense_outputs = [layer.output for layer in self.model.layers if isinstance(layer, tf.keras.layers.Dense)]
        # Exclude the last dense layer (output layer)
        hidden_outputs = dense_outputs[:-1]
        self.feature_extractor = tf.keras.Model(inputs=self.model.input, outputs=hidden_outputs)

    def on_epoch_end(self, epoch, logs=None):
        total_epochs = self.params.get('epochs', 100)
        # Aim to capture ~40-50 steps for smooth animations
        interval = max(1, total_epochs // 40)
        
        # Record on first epoch, final epoch, and at every interval
        if epoch % interval != 0 and epoch != total_epochs - 1:
            return

        # 1. Loss & Accuracy
        loss = logs.get('loss', 0.0)
        accuracy = logs.get('accuracy', 0.0)

        # 2. Decision boundary predictions
        preds = self.model(self.grid_points, training=False).numpy().flatten().tolist()

        # 3. Hidden layer representations (PCA projected to 2D)
        activations = self.feature_extractor(self.X, training=False)
        if not isinstance(activations, (list, tuple)):
            activations = [activations]
            
        hidden_representations = []
        for act_tensor in activations:
            act_np = act_tensor.numpy()
            n_neurons = act_np.shape[1]
            
            if n_neurons >= 2:
                try:
                    # Use PCA to project high-dimensional activations to 2D
                    pca = PCA(n_components=2, random_state=42)
                    act_2d = pca.fit_transform(act_np).tolist()
                except Exception:
                    # Fallback if PCA fails (e.g., zero variance)
                    act_2d = act_np[:, :2].tolist()
            else:
                # 1 neuron -> project onto 1D and pad with y=0 for scattering
                act_2d = [[float(val[0]), 0.0] for val in act_np]
            
            hidden_representations.append(act_2d)

        # 4. Weights and biases for connection plotting
        weights = []
        biases = []
        for layer in self.model.layers:
            if not isinstance(layer, tf.keras.layers.Dense):
                continue
            w, b = layer.get_weights()
            weights.append(w.tolist())
            biases.append(b.tolist())

        self.records.append({
            'epoch': epoch + 1,
            'loss': float(loss),
            'accuracy': float(accuracy),
            'decision_boundary': preds,
            'hidden_representations': hidden_representations,
            'weights': weights,
            'biases': biases
        })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train():
    data = request.get_json() or {}
    
    dataset_name = data.get('dataset_name', 'xor').lower()
    hidden_layers = data.get('layers', [4, 4])
    activation = data.get('activation', 'relu')
    learning_rate = data.get('learning_rate', 0.03)
    epochs = data.get('epochs', 150)
    noise = data.get('noise', 0.1)

    # Validate parameters
    epochs = min(max(int(epochs), 10), 1000)
    learning_rate = min(max(float(learning_rate), 0.0001), 1.0)
    
    # Convert layer configurations to ints
    hidden_layers = [min(max(int(n), 1), 20) for n in hidden_layers]
    if len(hidden_layers) == 0:
        hidden_layers = [4]
    elif len(hidden_layers) > 6:
        hidden_layers = hidden_layers[:6]

    # Generate data
    if dataset_name == 'xor':
        X, y = generate_xor(n_samples=200, noise=noise)
    elif dataset_name == 'spiral':
        X, y = generate_spiral(n_samples=200, noise=noise)
    elif dataset_name == 'moons':
        X, y = generate_moons(n_samples=200, noise=noise)
    elif dataset_name == 'circles':
        X, y = generate_circles(n_samples=200, noise=noise)
    else:
        return jsonify({'error': f'Unknown dataset: {dataset_name}'}), 400

    # Grid for decision boundary visualization
    grid_size = 30
    grid_x = np.linspace(-2.0, 2.0, grid_size)
    grid_y = np.linspace(-2.0, 2.0, grid_size)
    grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x, grid_y)
    grid_points = np.c_[grid_x_mesh.ravel(), grid_y_mesh.ravel()]

    # Clear tensorflow session to prevent memory leaks/graph clashes
    tf.keras.backend.clear_session()

    # Build MLP model using Functional API to ensure input/output layers are symbolics
    inputs = tf.keras.layers.Input(shape=(2,), name='input_layer')
    x = inputs
    for idx, neurons in enumerate(hidden_layers):
        x = tf.keras.layers.Dense(neurons, activation=activation, name=f'hidden_layer_{idx}')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Run training with custom history recorder callback
    recorder = HistoryRecorder(X, y, grid_points)
    model.fit(X, y, epochs=epochs, batch_size=32, callbacks=[recorder], verbose=0)

    # Extract final configuration details
    response_data = {
        'history': recorder.records,
        'grid': {
            'x': grid_x.tolist(),
            'y': grid_y.tolist(),
            'size': grid_size
        },
        'dataset': {
            'x1': X[:, 0].tolist(),
            'x2': X[:, 1].tolist(),
            'y': y.tolist()
        }
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

