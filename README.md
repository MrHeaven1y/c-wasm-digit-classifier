# 🧠 C → WebAssembly Digit Classifier

A handwritten digit classifier implemented in **pure C** and compiled to **WebAssembly**, running entirely in the browser with no backend or external ML frameworks.

> This project explores how low-level systems code can power real-time machine learning inference on the web.

---

## 🚀 Live Demo

👉 [C-Wasm-Digit-Classifier](https://webs-6t3b.onrender.com/)

<!-- 🔥 Add a GIF or screenshot here -->

<!-- ![Demo](assets/demo.gif) -->

Draw a digit and get instant predictions from a WebAssembly-powered C inference engine. 
**Note: It may be incorrect sometimes because of the lack of augmentation in training, since I solely focused on the training/inference.**

---

## ⚡ What makes this different

Most browser ML demos rely on high-level libraries.

This project deliberately avoids that.

* No TensorFlow.js
* No Python runtime in production
* No backend inference
* No abstractions hiding computation

Instead:

* 🧩 Inference implemented manually in C
* ⚙️ Compiled to WebAssembly using Emscripten
* 🌐 Runs fully client-side
* 🧠 Direct control over memory and execution

---

## 🏗️ System Architecture

```
[ Browser UI (Canvas + JS) ]
              │
              ▼
[ WebAssembly Module (Emscripten) ]
              │
              ▼
[ C Inference Engine ]
```

### Data Flow

1. User draws a digit on canvas
2. JavaScript extracts pixel data
3. Input is written into WebAssembly memory
4. C code performs forward inference
5. Result is returned to JavaScript

---

## 🧪 Model Pipeline (Training → Inference)

> **Note:** Training is performed offline.

* Model (e.g., simple MLP/CNN) trained using Python (e.g., PyTorch)
* Trained weights exported as static arrays
* Embedded directly into the C inference engine

This separation ensures:

* No runtime ML dependencies
* Deterministic execution
* Full control over inference pipeline

---

## 🧪 Core Components

### 🔹 Inference Engine (C)

* Manual forward pass implementation
* Hardcoded weights (from trained model)
* No external ML libraries
* Deterministic and memory-controlled execution

---

### 🔹 WebAssembly Bridge

* Built using Emscripten (`emcc`)
* Exposes C functions via `cwrap` / `ccall`
* Handles JS ↔ WASM memory interaction

---

### 🔹 Frontend (`docs/`)

* Canvas-based drawing interface
* Real-time prediction rendering
* Fully static (GitHub Pages compatible)

---

## ⚙️ Build System

Cross-platform build setup:

* `build.ps1` → Windows (PowerShell)
* `build.sh` → Linux/macOS

### Requirements

* Emscripten (`emcc`)
* Python 3 (for local server)

### Build

```bash
./build.sh
```

or (Windows):

```powershell
.\build.ps1
```

### Output

```
docs/
  ├── index.js
  ├── index.wasm
```

---

## 🌐 Run Locally

```bash
cd docs
python3 -m http.server
```

Open: http://localhost:8000

---

## 🌍 Deployment

* Hosted via GitHub Pages
* Served directly from `/docs`

---

## 📈 Performance Characteristics

* WebAssembly module size: ~<X> KB
* Inference latency: ~<Y> ms per prediction
* Near-native execution performance
* Zero network overhead (fully client-side)

---

## 🧠 Engineering Focus

This project is part of a broader direction toward:

* High-performance, low-level ML inference systems
* Deploying compute-heavy logic directly to the edge (browser)
* Bridging systems programming with modern web platforms
* Designing lightweight inference engines for real-time / embedded environments

---

## 🔮 Future Work

* [ ] External model loading (remove hardcoded weights)
* [ ] SIMD optimizations for faster inference
* [ ] WASM size reduction and load-time optimization
* [ ] Native vs WASM benchmarking
* [ ] Support for deeper architectures

---

## 👤 Author

**Dibyendu Mukherjee**
📧 [dibyendumukherjee916@gmail.com](mailto:dibyendumukherjee916@gmail.com)

---

## 📜 License

MIT License
