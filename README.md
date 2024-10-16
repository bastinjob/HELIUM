# Helium - A Tensor Compiler Using LLVM and MLIR

Helium is a tensor compiler built using LLVM and MLIR, designed to efficiently handle tensor operations for machine learning workloads and scientific computations. By leveraging MLIR’s extensible framework and LLVM’s powerful backend, Helium allows users to define tensor operations and generate optimized machine code.


## Introduction

Helium is a compiler that transforms high-level tensor operations into optimized machine code by using the following compilation pipeline:
- **Frontend**: Parsing tensor expressions.
- **Intermediate Representation (IR)**: MLIR-based representations with a custom tensor dialect.
- **Optimization**: Leverages MLIR passes and custom optimizations.
- **Backend**: Lowers to LLVM IR for machine code generation.

The goal of Helium is to provide high performance, extensibility, and an easy-to-use interface for tensor-heavy applications.

## Architecture

Helium is structured as a modular compiler system, leveraging both LLVM and MLIR frameworks. Below is a breakdown of its core components and stages:

### 1. **Frontend**: Parsing and Abstract Syntax Tree (AST) Generation
The frontend is responsible for taking high-level tensor expressions and transforming them into an intermediate representation (IR). This involves several steps:

- **Lexer**: Tokenizes the input tensor code by splitting it into meaningful symbols such as keywords, operators, and identifiers.
- **Parser**: Constructs an Abstract Syntax Tree (AST) from the tokens. This tree represents the syntactic structure of the program.
- **Semantic Analysis**: Ensures the correctness of the tensor operations (e.g., verifying dimensions of tensors and valid operations).

### 2. **Intermediate Representation (IR) - MLIR Tensor Dialect**
Helium utilizes MLIR (Multi-Level Intermediate Representation) to define a custom **Tensor Dialect**. MLIR provides multiple levels of abstraction to represent computations, and Helium takes advantage of this to implement tensor-specific optimizations. 

#### Tensor Dialect Operations:
- **`tensor.add`**: Represents element-wise addition of tensors.
- **`tensor.mul`**: Represents element-wise multiplication of tensors.
- **`tensor.matmul`**: Matrix multiplication operation.
- **`tensor.transpose`**: Transposes a tensor.
- **`tensor.reshape`**: Reshapes a tensor to a new dimension.
  
The Tensor Dialect acts as the backbone for the entire IR transformation pipeline, allowing Helium to apply various optimization and lowering techniques.

### 3. **Optimization Passes**
Helium’s optimizer applies a series of transformations and passes to improve the performance of tensor computations. These optimizations are designed to reduce computational complexity, memory usage, and ensure vectorization.

- **Constant Folding**: Simplifies constant expressions at compile-time.
- **Dead Code Elimination**: Removes unused or redundant code.
- **Common Subexpression Elimination**: Detects and eliminates duplicate calculations.
- **Loop Fusion**: Combines multiple loops that iterate over the same tensor data, improving cache performance and reducing memory traffic.

Helium leverages both custom MLIR-based optimizations and standard optimization passes provided by LLVM.

### 4. **Lowering to LLVM IR**
Once tensor operations are optimized, they are lowered to **LLVM IR** (Intermediate Representation). This lowering process translates the high-level tensor operations into scalar and vectorized operations that LLVM can further optimize and compile into machine code.

#### LLVM Backend Optimizations:
- **Loop Unrolling**: Expands loops to reduce loop overhead and improve instruction-level parallelism.
- **Vectorization**: Converts scalar operations into vector operations for SIMD (Single Instruction, Multiple Data) execution.
- **Inlining**: Replaces function calls with the function body to reduce function call overhead.

LLVM ensures that the final machine code is optimized for the target architecture, whether it be x86, ARM, or other platforms.

### 5. **Code Generation**
In the final stage, the lowered LLVM IR is translated into native machine code. Helium takes advantage of LLVM’s mature code generation framework to produce highly optimized executables for different architectures. This includes:
- **x86_64**: For standard desktop and server environments.
- **ARM**: For mobile and embedded systems.
- **RISCV**: Support for open-source hardware architectures.

### High-Level Flow Diagram



## Features
- **Tensor Dialect**: A custom MLIR dialect for tensor operations.
- **Optimizations**: Includes both MLIR-based and LLVM-based optimizations.
- **Extensibility**: Easily extend the compiler with new operations and transformations.
- **Multi-target Support**: Generates machine code for multiple architectures (x86, ARM, etc.).
- **Custom Command-Line Tools**: Tools to compile and run tensor programs.



### Prerequisites
- LLVM (>= 12.0)
- MLIR
- CMake (>= 3.13)
- Python 3 (for MLIR Python bindings)

### Steps to Install LLVM and MLIR
1. Clone the LLVM and MLIR repository:
   ```bash
   git clone https://github.com/llvm/llvm-project.git
   cd llvm-project
   mkdir build && cd build
   cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir"
   ninja


### 6. **Extensibility**
Helium is designed to be easily extensible. Developers can:
- Add new operations to the tensor dialect.
- Introduce new optimization passes at the MLIR level.
- Extend the lowering process to handle different hardware architectures.

### Summary of Key Components:
- **Frontend**: Parses tensor code and generates AST.
- **MLIR Dialect**: Custom Tensor dialect built on MLIR for flexible representation and optimization.
- **Optimization**: MLIR and LLVM optimization passes to improve tensor computations.
- **LLVM Backend**: Lowers MLIR to LLVM IR, and further optimizes for specific architectures.
- **Code Generation**: Outputs optimized machine code for execution on target hardware.

The modular design of Helium, powered by MLIR and LLVM, allows the tensor compiler to efficiently transform high-level operations into optimized low-level code, making it suitable for diverse applications in scientific computing and machine learning.