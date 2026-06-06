# 🧠 MNIST Neural Visualizer

[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org) [![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS) [![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/HTML) [![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript) ![GitHub repo size](https://img.shields.io/github/repo-size/Jalpan04/mnist-playground-visualizer) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A high-performance, real-time interactive neural network visualizer designed for the MNIST handwritten digit dataset. 

A high-performance, real-time interactive neural network visualizer designed for the MNIST handwritten digit dataset.

## ✨ Features

-   **Live Sketch-to-Inference**: Draw digits on the interactive canvas and watch the neural signals propagate through the layers in real-time.
-   **High-Performance Canvas Renderer**: Custom rendering engine utilizing path-batching and weight-culling to maintain 60 FPS while visualizing 50,000+ connections.
-   **Dynamic Activation Visuals**: Active neurons glow and illuminate their connections, providing immediate visual feedback of the "thought process" of the network.
-   **Real-time Training Stream**: Observe the network learning in the background with a live preview of the training data.
-   **Auto-pause UX**: Training automatically pauses during user drawing to ensure a clean, distraction-free inference experience.

## 🚀 Getting Started

### Prerequisites

-   [Node.js](https://nodejs.org/) (Project uses npm)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/tensorflow/playground.git
    cd playground
    ```

2.  **Install dependencies**:
    ```bash
    npm install
    ```

### Development

To run the visualizer locally with live-reloading:

```bash
npm run serve
```

To build for production:

```bash
npm run build
```

## 🛠 Tech Stack

-   **Frontend**: TypeScript, HTML5 Canvas, CSS3
-   **ML Engine**: Custom light-weight neural network library (TensorFlow Playground fork)
-   **Optimizations**: 
    -   **Path Batching**: Reduced draw calls by 2500x.
    -   **Weight Culling**: Ignored insignificant connections (<0.05) to save GPU/CPU cycles.
    -   **Glassmorphism**: Modern, premium UI styling.

## 📜 Credits

Based on the original [TensorFlow Playground](https://github.com/tensorflow/playground) by Google.
MNIST Data handling and real-time visualization enhancements by Antigravity.

---

*This is a high-performance educational tool for understanding deep learning activations. Not an official Google product.*
