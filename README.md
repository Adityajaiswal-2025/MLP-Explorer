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

---

## 🛠️ Architecture & Tech Stack

The application is split into a Python backend that handles TensorFlow model training/callback processing, and an interactive, highly responsive Javascript frontend.

*   **Backend:** Python 3, Flask, TensorFlow 2.x, NumPy, Scikit-Learn (for PCA feature projections).
*   **Frontend:** HTML5, CSS3 (Vanilla Glassmorphism & Cyberpunk glow themes), Javascript (ES6), Canvas API, Plotly.js (for high-fidelity plots).

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed on your system.

### 1. Clone & Set Up Directory
Navigate to the directory where you cloned the project:
```bash
cd "Deep Learning"
```

### 2. Install Dependencies
Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```
*(If you don't have a `requirements.txt` file, install them directly)*:
```bash
pip install Flask tensorflow numpy scikit-learn
```

### 3. Run the Server
Launch the Flask development server:
```bash
python app.py
```

### 4. Open in Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 📁 Repository Structure

```
├── app.py                # Flask server, dataset generators, and TensorFlow model training
├── templates/
│   └── index.html        # Main HTML layout, Tabs, Control panel, and educational content
├── static/
│   ├── css/
│   │   └── style.css     # Glassmorphic CSS design system, gradients, and custom scrollbars
│   └── js/
│       ├── api.js        # Handles Flask server communication
│       ├── network_visualizer.js # HTML5 Canvas rendering logic for the network topology
│       └── main.js       # Main application lifecycle, state management, and Plotly updates
└── .gitignore            # Files excluded from Git tracking
```

---

## 💡 How It Warps Space (The PCA feature)
A key highlight of **MLP Explorer** is the **Hidden Space Visualization**. For any chosen layer, the model captures the activation matrix $A \in \mathbb{R}^{N \times H}$ (where $N$ is the number of samples and $H$ is the number of hidden neurons in that layer). It applies Principal Component Analysis (PCA) to project this $H$-dimensional representation down to 2D. 

During training, you can watch the four quadrants of a XOR dataset twist, stretch, and line up until a straight line (the classifier) can cleanly cut through them.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
