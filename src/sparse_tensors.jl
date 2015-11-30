

export TensorOp, DiagonalOp, 
    IdentityOp, TensorOpProduct, 
    TensorOpProductKernel,
    create, destroy, sigma, share, shift, dim, tensor, 
    iidx, oidx, lidx,
    A_mul_B_generated!, make_mul_kernel, make_plus_mul_kernel

import Base: (*), transpose, conj, ctranspose, kron, A_mul_B!, sparse, full

abstract TensorOp

immutable DiagonalOp{T<:Number,J,K,M} <: TensorOp
    # T: data type
    # J: number of zeros before first non-zero
    # K: number of zeros after last non-zero
    # M: Shift offset
    # vals: diagonal elements between the non-zeros
    vals::Union{Vector{T},SharedVector{T}}
end


function share(a::SharedArray)
    a
end

function share(a)
    as = SharedArray(eltype(a),size(a)...)
    as[:] = a[:]
    as
end


function share{T,J,K,M}(op::DiagonalOp{T,J,K,M})
    DiagonalOp{T,J,K,M}(share(op.vals))
end

function shift{T,J,K,M}(::DiagonalOp{T,J,K,M})
    M
end

function dim{T,J,K,M}(d::DiagonalOp{T,J,K,M})
    J+length(d.vals)+K
end


function iidx{T,J,K,M}(d::DiagonalOp{T,J,K,M})
    # range of input indices that are processed in own dimension.
    (1:length(d.vals))+(J-M)
end

function oidx{T,J,K,M}(d::DiagonalOp{T,J,K,M})
    # range of output indices that are processed in own dimension.
    (1:length(d.vals))+J
end

function lidx{T,J,K,M}(d::DiagonalOp{T,J,K,M})
    # total length of processed input/output in own dimension.
    length(d.vals)
end

type IdentityOp{N} <: TensorOp
    # Identity operator with given dimension N
end

function share(op::IdentityOp)
    op
end


function shift{N}(::IdentityOp{N})
    0
end

function dim{N}(::IdentityOp{N})
    N
end

function iidx{N}(::IdentityOp{N})
    1:N
end

function oidx{N}(::IdentityOp{N})
    1:N
end

function lidx{N}(::IdentityOp{N})
    N
end

type TensorOpProduct
    # Product of single degree tensor ops
    ops::Vector{TensorOp}
    kernel
    TensorOpProduct(ops) = new(ops, nothing)
end

immutable TensorOpProductKernel{K}
    # Product of single degree tensor ops
    ops::NTuple{K,Union{DenseVector,IdentityOp}}
    iidx::NTuple{K,Range{Int64}}
    oidx::NTuple{K,Range{Int64}}
    lidx::NTuple{K, Int64}
end

function TensorOpProductKernel(tp::TensorOpProduct)
    iidxs = iidx(tp)
    oidxs = oidx(tp)
    lidxs = lidx(tp)
    ops_red = [(isa(op, DiagonalOp)? op.vals : op) for op in tp.ops]
    M = length(tp.ops)
    k = TensorOpProductKernel{M}(tuple(ops_red...), tuple(iidxs...), tuple(oidxs...), tuple(lidxs...))
    tp.kernel = k
    k
end

function apply_serial!(coeffs, toks::Vector{TensorOpProductKernel}, B::DenseArray, C::DenseArray, c=0)
    A_mul_B_generated!(coeffs[1], toks[1], B, c, C)
    for kk=2:length(toks)
        A_mul_B_generated!(coeffs[kk], toks[kk], B, 1, C)
    end
    C
end


function share(tp::TensorOpProduct)
    TensorOpProduct(map(share, tp.ops))
end

function share(tpk::TensorOpProductKernel)
    TensorOpProductKernel{length(tpk.ops)}(tuple(map(share, tpk.ops)), tpk.iidx, tpk.oidx, tpk.lidx) 
end


function shift(tp::TensorOpProduct)
    Int64[shift(op) for op=tp.ops]
end


for fn in [:iidx, :oidx, :lidx]
    @eval function $fn(tp::TensorOpProduct)
        map($fn, tp.ops)
    end
end

function ltot(tp::TensorOpProduct)
    prod(lidx(tp))
end


import Base: sparse, full


function sparse{N}(::IdentityOp{N})
    speye(N)
end

function sparse{T,J,K,M}(d::DiagonalOp{T,J,K,M})
    N = dim(d)
    spdiagm([zeros(T,J);d.vals; zeros(T,K)], 0,N,N) * spdiagm(ones(N-abs(M)),-M, N, N)
end
    
function full(d::TensorOp)
    full(sparse(d))
end


function sparse(tp::TensorOpProduct)
    reduce(kron, (SparseMatrixCSC)[sparse(op) for op in reverse(tp.ops)])
end

function full(tp::TensorOpProduct)
    full(sparse(tp))
end

function create(N::Int)
    DiagonalOp{Float64,1,0,1}(sqrt(1:(N-1)))
end

function destroy(N::Int)
    DiagonalOp{Float64,0,1,-1}(sqrt(1:(N-1)))
end
    
function sigma(N::Int,j::Int,k::Int)
    DiagonalOp{Float64,j-1,N-j,j-k}([1.])
end





function transpose(ii::IdentityOp)
    ii
end

function transpose{T,J,K,M}(d::DiagonalOp{T,J,K,M})
    DiagonalOp{T,J-M,K+M,-M}(d.vals)
end


function conj(ii::IdentityOp)
    ii
end

function conj{T,J,K,M}(d::DiagonalOp{T,J,K,M})
    if T <: Real 
        return d 
    end
    DiagonalOp{T,J,K,M}(conj(d.vals))
end


function ctranspose(op::TensorOp)
    transpose(conj(op))
end


for fn in [:transpose, :conj, :ctranspose]
    @eval function $fn(tp::TensorOpProduct)
        TensorOpProduct(TensorOp[$fn(op) for op in tp.ops])
    end
end

function (*){N,M}(a::IdentityOp{N}, b::IdentityOp{M})
    N == M || error("Wrong shape for multiplication: ($N x $N) * ($M x $M)")
    b
end

function (*){N}(a::IdentityOp{N}, b::TensorOp)
    M = dim(b)
    N == M || error("Wrong shape for multiplication: ($N x $N) * ($M x $M)")
    b
end

function (*){M}(a::TensorOp, b::IdentityOp{M})
    N = dim(a)
    N == M || error("Wrong shape for multiplication: ($N x $N) * ($M x $M)")
    a
end


function (*){S,T,J1,K1,M1,J2,K2,M2}(a::DiagonalOp{S,J1,K1,M1}, b::DiagonalOp{T,J2,K2,M2})
    N, M = dim(a), dim(b)
    N == M || error("Wrong shape for multiplication: ($N x $N) * ($M x $M)")
    da = [zeros(S, J1); a.vals; zeros(S,K1)]
    db = [zeros(T, J2); b.vals; zeros(T,K2)]
    if M1 >= 0
        dbshifted = [zeros(T, M1); db[1:(M-M1)]]
    else
        dbshifted = [db[(abs(M1)+1):M];zeros(T, abs(M1))]
    end
    dab = da .* dbshifted
    q, r = find(abs(dab) .> 0)[[1,end]]
    J = q - 1
    K = N - r
    DiagonalOp{eltype(dab),J,K,M1+M2}(dab[q:r])
end

function tensor(a::TensorOp, b::TensorOp...)
    TensorOpProduct([a; b...])
end

function tensor(a::TensorOp, b::TensorOpProduct)
    TensorOpProduct([a; b.ops])
end

function tensor(a::TensorOpProduct, b::TensorOp)
    TensorOpProduct([a.ops; b])
end

TensorType = Union{TensorOp, TensorOpProduct}

function kron(a::TensorType, b::TensorType...)
    tensor(reverse(b)...,a)
end


function A_mul_B!{T,N}(C::DenseArray{T,N}, A::TensorOpProduct, B::DenseArray{T,N})
    N == length(A.ops) || error("Mismatch between tensor factors")
    size(B) === size(C) || error("Dimension mismatch between input and output")
    iidxs = iidx(A)
    oidxs = oidx(A)
    reshaped_ops = []
    for (k,o)=enumerate(A.ops)
        if !isa(o, IdentityOp)
            push!(reshaped_ops, reshape(o.vals, [ones(Int64,k-1);lidx(o); ones(Int64,N-k)]...))
        end
    end
    broadcast!(*, sub(C, oidxs...), sub(B, iidxs...), reshaped_ops...)
    C
end

function make_mul_kernel(A::TensorOpProduct)
    M = length(A.ops)
    iidxs = iidx(A)
    oidxs = oidx(A)
    reshaped_ops = []
    for (k,o)=enumerate(A.ops)
        if !isa(o, IdentityOp)
            push!(reshaped_ops, reshape(o.vals, [ones(Int64,k-1);lidx(o); ones(Int64,M-k)]...))
        end
    end

    function kernel!{T,N}(C::DenseArray{T,N}, B::DenseArray{T,N})
        N == length(A.ops) || error("Mismatch between tensor factors")
        size(B) === size(C) || error("Dimension mismatch between input and output")
        broadcast!(*, sub(C, oidxs...), sub(B, iidxs...), reshaped_ops...)
        C
    end
end    


    

function plus_mul(a, b...)
    a + (*)(b...)
end

const b_plus_mul! = broadcast!_function(plus_mul)


function make_plus_mul_kernel(T::DataType, A::TensorOpProduct)
    mul_kernel! = make_mul_kernel(A)

    M = length(A.ops)
    iidxs = iidx(A)
    oidxs = oidx(A)
    reshaped_ops = []
    for (k,o)=enumerate(A.ops)
        if !isa(o, IdentityOp)
            push!(reshaped_ops, reshape(o.vals, [ones(Int64,k-1);lidx(o); ones(Int64,M-k)]...))
        end
    end
    push!(reshaped_ops, reshape(collect(one(T)),ones(Int64,M)...))

    function kernel!{S,N}(a::T, B::DenseArray{S,N}, c::S, C::DenseArray{S,N})
        N == length(A.ops) || error("Mismatch between tensor factors")
        size(B) === size(C) || error("Dimension mismatch between input and output")

        if c != one(S)
            if c == zero(S)
                fill!(C, c)
            else
                scale!(C, c)
            end
        end

        if a == zero(T)
            return C
        elseif a == one(T)
            return mul_kernel!(C, B)
        end
        reshaped_ops[end][1] = a

        b_plus_mul!(sub(C, oidxs...), sub(C, oidxs...), sub(B, iidxs...), reshaped_ops...)
        C
    end
end    



function A_mul_B!{T,N}(a::T, A::TensorOpProduct, B::DenseArray{T,N}, c::T, C::DenseArray{T,N})

    N == length(A.ops) || error("Mismatch between tensor factors")
    size(B) === size(C) || error("Dimension mismatch between input and output")

    if c != one(T)
        if c == zero(T)
            fill!(C, c)
        else
            scale!(C, c)
        end
    end

    if a == zero(T)
        return C
    end

    iidxs = iidx(A)
    oidxs = oidx(A)

    reshaped_ops = []
    for (k,o)=enumerate(A.ops)
        if !isa(o, IdentityOp)
            push!(reshaped_ops, reshape(o.vals, [ones(Int64,k-1);lidx(o); ones(Int64,N-k)]...))
        end
    end
    push!(reshaped_ops, reshape(collect(a), ones(Int64, N)...))
    b_plus_mul!(sub(C, oidxs...), sub(C, oidxs...), sub(B, iidxs...), reshaped_ops...)
    C
end



function A_mul_B_generated!(a, A::TensorOpProduct, B::DenseArray, c, C::DenseArray)
    k = A.kernel === nothing ? TensorOpProductKernel(A) : A.kernel
    A_mul_B_generated!(a, k, B, c, C)
end

function A_mul_B_generated!{K}(a, A::TensorOpProductKernel{K}, B::DenseArray, c, C::DenseArray)
    ndims(B) == K || error("Mismatch between tensor factors")
    size(B) === size(C) || error("Dimension mismatch between input and output")
    top_kernel!(c, sub(C, A.oidx...), sub(B, A.iidx...), a, A.ops...)
    C
end



@generated function top_kernel!(c, sC, sB, a, Avs::Union{DenseVector,IdentityOp}...)
   code = top_kernel!_code(c, sC, sB, a, Avs)
   println(code)
   code
end


function top_kernel!_code(c, sC, sB, a, Avs)
    K = length(Avs)
    @assert K >= 1
    loop_vars = [symbol("k_$(j)") for j=1:K]
    sCref = Expr(:ref, :sC, loop_vars...)
    sBref = Expr(:ref, :sB, loop_vars...)
    Avrefs = []
    for kk=1:K
        if Avs[kk] <: DenseArray
            push!(Avrefs, Expr(:ref, Expr(:ref, :Avs, kk), loop_vars[kk]))
        else
            @assert Avs[kk] <: IdentityOp 
        end
    end

    innerloop_prod = Expr(:call, :*, :a, sBref, Avrefs...)
    innerloop = quote
        @simd for $(loop_vars[1]) = 1:size(sC,1)
            @inbounds $sCref = c * $sCref + $innerloop_prod
        end
    end
    loop = innerloop
    for kk=2:K
        loop = quote 
            for $(loop_vars[kk]) in 1:size(sC,$kk)
                $loop
            end
        end
    end
    loop
end




function tensor{S,T,M,N}(A::Array{T,M}, B::Array{S,N})
    broadcast(*, reshape(A, size(A)...,ones(Int64,N)...),reshape(B,ones(Int64,M)...,size(B)...))
end



