

export TensorOp, DiagonalOp, 
    IdentityOp, TensorOpProduct, 
    TensorOpProductKernel,
    create, destroy, sigma, 
    share, dim, tensor, 
    iidx, oidx

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


type IdentityOp{N} <: TensorOp
    # Identity operator with given dimension N
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
end

function TensorOpProductKernel(tp::TensorOpProduct)
    iidxs = iidx(tp)
    oidxs = oidx(tp)
    ops_red = [(isa(op, DiagonalOp)? op.vals : op) for op in tp.ops]
    M = length(tp.ops)
    k = TensorOpProductKernel{M}(tuple(ops_red...), tuple(iidxs...), tuple(oidxs...))
    tp.kernel = k
    k
end

function share(a::SharedArray)
    a
end

function share(a::Array)
    as = SharedArray(eltype(a),size(a)...)
    as[:] = a[:]
    as
end

function share(op::IdentityOp)
    op
end


function share{T,J,K,M}(op::DiagonalOp{T,J,K,M})
    DiagonalOp{T,J,K,M}(share(op.vals))
end


function share(tp::TensorOpProduct)
    TensorOpProduct(map(share, tp.ops))
end

function share(tpk::TensorOpProductKernel)
    TensorOpProductKernel{length(tpk.ops)}(tuple(map(share, tpk.ops)), tpk.iidx, tpk.oidx) 
end




function dim{T,J,K,M}(d::DiagonalOp{T,J,K,M})
    J+length(d.vals)+K
end

function dim{N}(::IdentityOp{N})
    N
end



# range of input indices that are processed in own dimension.
iidx{T,J,K,M}(d::DiagonalOp{T,J,K,M}) = (1:length(d.vals))+(J-M)
iidx{N}(::IdentityOp{N}) = 1:N

# range of output indices that are processed in own dimension.
oidx{T,J,K,M}(d::DiagonalOp{T,J,K,M}) =  (1:length(d.vals))+J
oidx{N}(::IdentityOp{N})=1:N


for fn in [:iidx, :oidx]
    @eval $fn(tp::TensorOpProduct) = map($fn, tp.ops)
end


sparse{N}(::IdentityOp{N})= speye(N)

function sparse{T,J,K,M}(d::DiagonalOp{T,J,K,M})
    N = dim(d)
    spdiagm([zeros(T,J);d.vals; zeros(T,K)], 0,N,N) * spdiagm(ones(N-abs(M)),-M, N, N)
end
sparse(tp::TensorOpProduct) = reduce(kron, (SparseMatrixCSC)[sparse(op) for op in reverse(tp.ops)])
    
full(d::Union{TensorOp, TensorOpProduct}) = full(sparse(d))


create(N::Int) = DiagonalOp{Float64,1,0,1}(sqrt(1:(N-1)))
destroy(N::Int) = DiagonalOp{Float64,0,1,-1}(sqrt(1:(N-1)))
sigma(N::Int,j::Int,k::Int) = DiagonalOp{Float64,j-1,N-j,j-k}([1.])

transpose(ii::IdentityOp) = ii
transpose{T,J,K,M}(d::DiagonalOp{T,J,K,M}) =  DiagonalOp{T,J-M,K+M,-M}(d.vals)
conj(ii::IdentityOp) = ii
conj{T,J,K,M}(d::DiagonalOp{T,J,K,M}) = DiagonalOp{T,J,K,M}(conj(d.vals))
ctranspose(op::TensorOp)= transpose(conj(op))

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

function tensor{S,T,M,N}(A::Array{T,M}, B::Array{S,N})
    broadcast(*, reshape(A, size(A)...,ones(Int64,N)...),reshape(B,ones(Int64,M)...,size(B)...))
end



# In-place mat-vec product without coefficients
A_mul_B!(C::DenseArray, A::Union{TensorOpProduct,TensorOpProductKernel}, B::DenseArray) = A_mul_B!(1, A, B, 0, C)

function A_mul_B!(a, A::TensorOpProduct, B::DenseArray, c, C::DenseArray)
    k = A.kernel === nothing ? TensorOpProductKernel(A) : A.kernel
    A_mul_B!(a, k, B, c, C)
end

function A_mul_B!{K}(a, A::TensorOpProductKernel{K}, B::DenseArray, c, C::DenseArray)
    ndims(B) == K || error("Mismatch between tensor factors")
    size(B) === size(C) || error("Dimension mismatch between input and output")
    top_kernel!(c, sub(C, A.oidx...), sub(B, A.iidx...), a, A.ops...)
    C
end

# code for generating mat-vec loops
@generated function top_kernel!(c, sC, sB, a, Avs::Union{DenseVector,IdentityOp}...)
   code = top_kernel!_code(c, sC, sB, a, Avs)
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



function apply_serial!(coeffs, toks::Vector{TensorOpProductKernel}, B::DenseArray, C::DenseArray, c=0)
    A_mul_B_generated!(coeffs[1], toks[1], B, c, C)
    for kk=2:length(toks)
        A_mul_B_generated!(coeffs[kk], toks[kk], B, 1, C)
    end
    C
end
