# QuantumTensors

[![Build Status](https://travis-ci.org/ntezak/QuantumTensors.jl.svg?branch=master)](https://travis-ci.org/ntezak/QuantumTensors.jl)


This package contains special methods for efficiently representing sparse local quantum operators that act on a tensor product space.
The most important methods are:

    using QuantumTensors

    # Create a creation operator with N basis states
    create(N)

    # Create an annihilation operator with N basis states
    destroy(N)

    # Create an angular momentum J-z operator
    jz(N)

    # Create a truncated angular momentum J_z operator with only K basis states
    jz(J, K)

    # Create an angular momentum J_+ operator (with optional basis truncation)
    jplus(J, K)

    # Create an angular momentum J_- operator (with optional basis truncation)
    jminus(J, K)


    # products within a local degree of freedom
    jz(J,K) * jplus(J,K)

    # form a tensor product between different degrees of freedom
    tensor(jz(J), create(N))


    # in-place apply a single tensor product operator to a state
    # (needs to be an array with as many dimensions as the tensor product factors)
    A_mul_B!(λ, A, Ψ, μ, Φ) ==>  Φ -> μ Φ + λ A.Ψ


    # in-place apply a sequence single tensor product operators to a state
    # with certain coefficients (supports CSR type sparse matrices as well)
    apply_serial!([λ1,λ2], [A1,A2], Ψ, μ, Φ) ==>  Φ -> μ Φ + λ1 A1.Ψ + λ2 A2.Ψ


Documentation will fleshed out more soon. See the tests for some further usage examples.
