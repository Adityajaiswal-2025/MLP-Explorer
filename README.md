# 🧠 MLP Explorer: Representation Power of Multilayer Perceptrons

**MLP Explorer** is an interactive, web-based educational hub and playground designed to visualize and demystify the internal workings of Multilayer Perceptrons (MLPs). It demonstrates how neural networks perform coordinate transformations, bend space, learn representations, and approximate complex functions.

Designed with premium, glassmorphic UI aesthetics and rich real-time visualizations, it acts as a localized, extended version of TensorFlow Playground with advanced features like **Hidden Layer Activation PCA Projections** and **Side-by-Side Model Comparison**.

---

## 🌟 Key Features

### 🎮 1. Interactive Playground
*   **Dynamic Data Generation:** Choose from four classical datasets: XOR, Moons, Concentric Circles, and Interleaved Spirals, and adjust noise levels on the fly.
*   **Hyperparameter Tuning:** Fine-tune learning rates, select activations (ReLU, Sigmoid, Tanh), and control training epochs.
*   **Custom Topology Builder:** Construct networks with up to 5 hidden layers and custom neuron allocations per layer.
*   **Interactive Playback Timeline:** Pause, step frame-by-frame, scrub the timeline, or control speed to analyze exactly *how* the network converges epoch-by-epoch.

### 📊 2. Real-Time Visualizations
*   **Decision Boundary Plot:** Interactive 2D classification boundary updates as the network trains.
*   **Network Architecture Canvas:** Animated, live-rendering HTML5 canvas showing neuron activations, positive/negative weights (represented by color and thickness), and signal feedforward pulses.
*   **Feature Learning (Hidden Space):** Projects high-dimensional hidden layer activations into 2D using Principal Component Analysis (PCA). Watch in real-time how the network warps coordinate space to make complex boundaries linearly separable!
*   **Convergence Analytics:** Real-time curves tracking training loss and classification accuracy.

### ⚖️ 3. Side-by-Side Comparison Mode
*   Train two different architectures (e.g., a shallow/narrow network vs. a deep/wide network) simultaneously.
*   Compare decision boundaries, final loss, accuracy, and latent spaces side-by-side on the exact same dataset to understand model capacity, underfitting, and overfitting.

### 📖 4. Educational Hub
*   Includes built-in interactive articles on MLP mathematical models, non-linearities, space warping, the Universal Approximation Theorem, and network capacity.

