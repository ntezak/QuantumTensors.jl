

export TensorOp, DiagonalOp, 
    IdentityOp, TensorOpProduct, 
    TensorOpProductKernel,
    create, destroy, sigma, 
    share, dim, tensor, 
    iidx, oidx

import Base: (*), transpose, conj, ctranspose, kron, A_mul_B!, sparse, full, 
              convert, promote_rule

abstract TensorOp

immutable DiagonalOp{T<:Number} <: TensorOp
    # T: data type
    # J: number of zeros before first non-zero
    # K: number of zeros after last non-zero
    # M: Shift offset
    # vals: diagonal elements between the non-zeros
    vals::Vector{T}
    jkm::NTuple{3,Int64}
    # expr::Expr
    # DiagonalOp{T}(vals::Vector{T}, jkm, expr = :()) = new{T}(vals, jkm, expr)
end

# function DiagonalOp(T, op::DiagonalOp)
#   TT = typeof(op)
#   J,K,M = TT.parameters[2:end]
#   DiagonalOp{T,J,K,M}(convert(Vector{T}, op.vals))
# end
#

promote_rule{T,S}(::Type{DiagonalOp{T}}, ::Type{DiagonalOp{S}}) = DiagonalOp{promote_type(T,S)}
convert{T}(::Type{DiagonalOp{T}}, op::DiagonalOp) = DiagonalOp(convert(Vector{T}, op.vals), op.jkm)


type IdentityOp{N} <: TensorOp
    # Identity operator with given dimension N
end

type TensorOpProduct{T,K}
    # Product of single degree tensor ops
    ops::NTuple{K,Union{DiagonalOp{T},IdentityOp}}
    kernel
end
TensorOpProduct(ops::TensorOp...) = TensorOpProduct(promote_diags(ops...), nothing)


function promote_diags(ops::TensorOp...)
  opsv = collect(ops)
  nontrivial=[isa(op, DiagonalOp) for op=opsv]
  promoted_opsv = Array(Any,length(ops))
  promoted_opsv[nontrivial] = collect(promote(opsv[nontrivial]...))
  promoted_opsv[~nontrivial] = opsv[~nontrivial]
  tuple(promoted_opsv...)
end

function convert_diags{T}(S::Type{DiagonalOp{T}}, ops::TensorOp...)
  nops = Any[]
  for op in ops
    if isa(op, DiagonalOp)
      push!(nops, convert(S, op))
    else
      push!(nops, op)
    end
  end
  tuple(nops...)
end



promote_rule{T,S,K}(::Type{TensorOpProduct{T,K}}, ::Type{TensorOpProduct{S,K}}) = TensorOpProduct{promote_type(T,S),K}
promote_rule{T,S}(::Type{TensorOpProduct{T}}, ::Type{TensorOpProduct{S}}) = TensorOpProduct{promote_type(T,S)}
convert{T,S,K}(::Type{TensorOpProduct{T,K}}, top::TensorOpProduct{S,K}) = TensorOpProduct(convert_diags(DiagonalOp{T}, top.ops...)...)
convert{T}(::Type{TensorOpProduct{T}}, top::TensorOpProduct) = TensorOpProduct(convert_diags(DiagonalOp{T}, top.ops...)...)


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

type TensorOpProductSum{T,K,N}
    # Sum of product of single degree tensor ops
    tops::NTuple{N,NTuple{K,Union{IdentityOp,DiagonalOp{T}}}}
    kernel
end

function TensorOpProductSum{K}(tops::TensorOpProduct{K}...)
  tops = promote(tops...)
  topops = [top.ops for top in tops]
  println(topops)
  TensorOpProductSum(tuple(topops...), nothing)
end

@generated function apply_topsum!(ts, tops, B::DenseArray, c, C::DenseArray)
  println(ts, tops, B, c, C)
  nothing
end

      

# 
# function share(a::SharedArray)
#     a
# end
# 
# function share(a::Array)
#     as = SharedArray(eltype(a),size(a)...)
#     as[:] = a[:]
#     as
# end
# 
# function share(op::IdentityOp)
#     op
# end
# 
# 
# function share{T,J,K,M}(op::DiagonalOp{T,J,K,M})
#     DiagonalOp{T,J,K,M}(share(op.vals))
# end
# 
# 
# function share(tp::TensorOpProduct)
#     TensorOpProduct(map(share, tp.ops))
# end
# 
# function share(tpk::TensorOpProductKernel)
#     TensorOpProductKernel{length(tpk.ops)}(tuple(map(share, tpk.ops)), tpk.iidx, tpk.oidx) 
# end
# 

function dim(d::DiagonalOp)
  J,K,M = d.jkm
  J+length(d.vals)+K
end

function dim{N}(::IdentityOp{N})
    N
end



# range of input indices that are processed in own dimension.
iidx(d::DiagonalOp) = (1:length(d.vals))+(d.jkm[1]-d.jkm[3])
iidx{N}(::IdentityOp{N}) = 1:N

# range of output indices that are processed in own dimension.
oidx(d::DiagonalOp) =  (1:length(d.vals))+d.jkm[1]
oidx{N}(::IdentityOp{N})=1:N


for fn in [:iidx, :oidx]
    @eval $fn(tp::TensorOpProduct) = map($fn, tp.ops)
end


# conversion to sparse matrices
sparse{N}(::IdentityOp{N})= speye(N)
function sparse{T}(d::DiagonalOp{T})
    N = dim(d)
    J,K,M = d.jkm
    spdiagm([zeros(T,J);d.vals; zeros(T,K)], 0,N,N) * spdiagm(ones(N-abs(M)),-M, N, N)
end
sparse(tp::TensorOpProduct) = reduce(kron, (SparseMatrixCSC)[sparse(op) for op in reverse(tp.ops)])

# conversion of full matrices
full(d::Union{TensorOp, TensorOpProduct}) = d |> sparse |> full

create(N::Int) = DiagonalOp{Float64}(sqrt(1:(N-1)),(1,0,1))
destroy(N::Int) = DiagonalOp{Float64}(sqrt(1:(N-1)),(0,1,-1))
sigma(N::Int,j::Int,k::Int) = DiagonalOp{Float64}([1.],(j-1,N-j,j-k))

transpose(ii::IdentityOp) = ii
transpose(d::DiagonalOp) =  DiagonalOp(d.vals,(d.jkm[1]-d.jkm[3],d.jkm[2]+d.jkm[3],-d.jkm[3]))
conj(ii::IdentityOp) = ii
conj(d::DiagonalOp) = DiagonalOp(conj(d.vals), d.jkm)
ctranspose(op::TensorOp)= transpose(conj(op))

for fn in [:transpose, :conj, :ctranspose]
    @eval function $fn(tp::TensorOpProduct)
        TensorOpProduct([$fn(op) for op in tp.ops]...)
    end
end


# Multiplication of diagonal tensors on same degree of freedom
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

function (*){S,T}(a::DiagonalOp{S}, b::DiagonalOp{T})
    N, M = dim(a), dim(b)
    J1,K1,M1 = a.jkm
    J2,K2,M2 = b.jkm
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
    DiagonalOp{eltype(dab)}(dab[q:r],(J,K,M1+M2))
end


# construction of tensor products
function tensor(a::TensorOp...)
    TensorOpProduct(a...)
end

function tensor(a::TensorOp, b::TensorOpProduct)
    TensorOpProduct(a, b.ops...)
end

function tensor(a::TensorOpProduct, b::TensorOp)
    TensorOpProduct(a.ops..., b)
end

# construction of tensor products for states
function tensor{S,T,M,N}(A::Array{T,M}, B::Array{S,N})
    broadcast(*, reshape(A, size(A)...,ones(Int64,N)...),reshape(B,ones(Int64,M)...,size(B)...))
end


# In-place mat-vec product without coefficients
A_mul_B!(C::DenseArray, A::Union{TensorOpProduct,TensorOpProductKernel}, B::DenseArray) = A_mul_B!(1, A, B, 0, C)

# In-place mat-vec product with coefficients
function A_mul_B!(a, A::TensorOpProduct, B::DenseArray, c, C::DenseArray)
    k = A.kernel === nothing ? TensorOpProductKernel(A) : A.kernel
    A_mul_B!(a, k, B, c, C)
end

function A_mul_B!{K}(a, A::TensorOpProductKernel{K}, B::DenseArray, c, C::DenseArray)
    ndims(B) == K || error("Mismatch between tensor factors")
    size(B) === size(C) || error("Dimension mismatch between input and output")
    if c != 1
      if c == 0
        fill!(C, 0)
      else
        scale!(C, c)
      end
    end
    top_kernel!(sub(C, A.oidx...), sub(B, A.iidx...), a, A.ops...)
    C
end

# code for generating mat-vec loops
@generated function top_kernel!(sC, sB, a, Avs::Union{DenseVector,IdentityOp}...)
   code = top_kernel_code(sC, sB, a, Avs)
   println(code)
   code
end
function top_kernel_code(sC, sB, a, Avs)
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
            @inbounds $sCref += $innerloop_prod
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
    A_mul_B!(coeffs[1], toks[1], B, c, C)
    for kk=2:length(toks)
        A_mul_B!(coeffs[kk], toks[kk], B, 1, C)
    end
    C
end
