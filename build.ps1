$ErrorActionPreference = "Stop"

# ---------------------------
# Config
# ---------------------------
$SRC_DIR = "src"
$INCLUDE_DIR = "include"
$OUT_DIR = "docs"
$OUTPUT_NAME = "index"

# ---------------------------
# Ensure output dir exists
# ---------------------------
if (!(Test-Path $OUT_DIR)) {
    New-Item -ItemType Directory -Path $OUT_DIR | Out-Null
}

Write-Host "Building WebAssembly module..."

# ---------------------------
# Build with Emscripten (Single line to prevent formatting errors)
# ---------------------------
emcc "$SRC_DIR/wasm_main.c" "$SRC_DIR/tensor.c" -I"$INCLUDE_DIR" -o "$OUT_DIR/$OUTPUT_NAME.js" -s WASM=1 -s MODULARIZE=1 -s EXPORT_NAME="Module" -s EXPORTED_FUNCTIONS="['_predict', '_init_system', '_malloc', '_free']" -s EXPORTED_RUNTIME_METHODS="['cwrap','ccall']" -s ALLOW_MEMORY_GROWTH=1 -O3

Write-Host "Build complete:"
Write-Host "   → $OUT_DIR/$OUTPUT_NAME.js"
Write-Host "   → $OUT_DIR/$OUTPUT_NAME.wasm"