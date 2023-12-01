# Dignostics-labs
in this branch we will look to improve the first version by building it using this techstack 
Rust
All our backend services and all our data processing is implemented in Rust.  it's the optimal choice for efficient, safe, and scalable applications. It also offers effortless interoperability with Python.

JAX & XLA
Our neural networks are implemented in JAX, and we have a number of custom XLA operations to improve their efficiency.

Triton & CUDA
Running our large neural networks at scale while maximizing compute efficiency is paramount if we want to make optimal use of our compute resources. Hence, we regularly write bespoke kernels either in Triton or in raw C++ CUDA. (we could find other efficient ways of doing it)

TypeScript, React & Angular
Our frontend code is written exclusively in TypeScript using either React or Angular. Backend communication is facilitated by gRPC-web APIs for type safety.
