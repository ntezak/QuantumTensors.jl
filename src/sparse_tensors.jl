

export TensorOp, DiagonalOp, 
    IdentityOp, TensorOpProduct, 
    TensorOpProductSum,
    TensorOpProductKernel,
    create, destroy, sigma, jz, jplus, jminus,
    share, dim, tensor, 
    iidx, oidx, all_close, apply_serial!

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

type IdentityOp{N} <: TensorOp
  # Identity operator with given dimension N
end

promote_rule{T,S}(::Type{DiagonalOp{T}}, ::Type{DiagonalOp{S}}) = DiagonalOp{promote_type(T,S)}
convert{T}(::Type{DiagonalOp{T}}, op::DiagonalOp) = DiagonalOp(convert(Vector{T}, op.vals), op.jkm)
convert{T,N}(::Type{DiagonalOp{T}}, op::IdentityOp{N}) = DiagonalOp(ones(T, N), (0,0,0))


function all_close(op1::DiagonalOp, op2::DiagonalOp; tol=1e-8)
  dim(op1) == dim(op2) || error("Incompatible shapes")
  op1.jkm == op2.jkm && norm(op1.vals-op2.vals) < tol
end

all_close{T}(op1::IdentityOp, op2::DiagonalOp{T}; kw...) = all_close(convert(DiagonalOp{T},op1),op2; kw...)
all_close{N}(op1::IdentityOp{N}, op2::IdentityOp{N}; kw...) = true
all_close{T}(op1::DiagonalOp{T}, op2::IdentityOp; kw...) = all_close(convert(DiagonalOp{T},op2),op1; kw...)



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


all_close(top1::TensorOpProduct, top2::TensorOpProduct; tol=1e-8) = all(Bool[all_close(op1, op2; tol=tol) for (op1,op2) in zip(top1.ops, top2.ops)])



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

function TensorOpProductSum(tops::TensorOpProduct...)
  tops = promote(tops...)
  topops = [top.ops for top in tops]
  TensorOpProductSum(tuple(topops...), nothing)
end

all_close{T,S,K,N}(tops1::TensorOpProductSum{T,K,N}, 
          tops2::TensorOpProductSum{S,K,N}; tol=1e-8) = all(
            Bool[all_close(tops1.tops[n][k], tops2.tops[n][k]; tol=tol) for n=1:N, k=1:K])


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

function jz(J::Int,trunc=-1)
  if trunc > 0
    N = trunc
  else
    N = 2J
  end
  DiagonalOp{Float64}(1.*collect(-J:(-J+N)),(0,0,0))
end

function jpm(J::Int, trunc=-1, pm=:(+))
  if trunc > 0
    N = trunc
  else
    N = 2J
  end
  rng = -J:(-J+N-1)
  diag = sqrt(J*(J+1) - rng .* (rng+1))
  if pm === :(+)
    DiagonalOp{Float64}(diag, (1,0,1))
  elseif pm === :(-)
    DiagonalOp{Float64}(diag, (0,1,-1))
  else
    error("Can only create J_+ and J_-")
  end
end

jplus(J, trunc=-1) = jpm(J,trunc, :(+))
jminus(J, trunc=-1) = jpm(J,trunc, :(-))
  

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
  #  println(code)
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





function apply_serial!(coeffs, toks, B, C, c=0)
    A_mul_B!(coeffs[1], toks[1], B, c, C)
    for kk=2:length(toks)
        A_mul_B!(coeffs[kk], toks[kk], B, 1, C)
    end
    C
end

function apply_serial!(coeffs, toks::TensorOpProductSum, B::DenseArray, C::DenseArray, c=0)
    apply_topsum!(coeffs, toks.tops, B, c, C)
end

@generated function apply_topsum!{T,N,K}(ts, tops::NTuple{N,NTuple{K,Union{IdentityOp,DiagonalOp{T}}}}, B::DenseArray{T,K}, c, C::DenseArray{T,K})
  code = apply_topsum_code(ts, tops, B, c, C)
    # println(code)
    code
end

function apply_topsum_code(ts, tops, B, c, C)
    # function diag_op_code(has_identity, offsets::Matrix{Int64}, nontrivial_factors::Matrix{Bool})
    # N = number of nontrivial terms in sum
    # D = number of tensor dimensions.
    N = length(tops.parameters)
    N >= 1 || error("Trivial Operator sum passed.")
    D = length(tops.parameters[1].parameters)
    
    nontrivial_factors = Array(Bool, N, D)
    for n=1:N
        for d=1:D
            nontrivial_factors[n,d] = tops.parameters[n].parameters[d] <: DiagonalOp
        end
    end
    

    
    IIDXs = [symbol("IIDX_$(n)") for n=1:N]
    kds = [symbol("k_$d") for d=1:D]
    flags = [symbol("apply_t$(n)_d$(d)") for n=1:N, d=1:D]
    
    diags = [symbol("diag_t$(n)_d$(d)") for n=1:N, d=1:D]
    jnds = [symbol("j_$(n)_$(d)") for n=1:N, d=1:D]
    knds = [symbol("k_$(n)_$(d)") for n=1:N, d=1:D]
    mnds = [symbol("m_$(n)_$(d)") for n=1:N, d=1:D]
    kdsmmnds = [nontrivial_factors[n,d] ? :($(kds[d])-$(mnds[n,d])) : kds[d] for n=1:N, d=1:D]
    
    diag_assignments = quote
        # Assign DiagonalOp.vals and .jkm properties to variables
    end
    for n=1:N
        for d=1:D
            if nontrivial_factors[n,d]
                push!(diag_assignments.args, quote
                    $(diags[n,d]) = tops[$n][$d].vals
                    ($(jnds[n,d]),$(knds[n,d]),$(mnds[n,d])) = tops[$n][$d].jkm
#                     println($n, ", ", $d, ": ", tops[$n][$d].jkm)
                    end)
            end
        end
    end
    
    iidx_init = quote
        IIDX=1
    end
    
    iidx_inc = quote
        IIDX += 1
    end
    
    for n=1:N
        terms = Any[]
        for d=1:D
            if nontrivial_factors[n,d]
                push!(terms, :($(mnds[n,d]) * stride(C,$d)))
            end
        end
        if length(terms) > 1
            offset = :(1-$(Expr(:call, :+, terms...)))
        elseif length(terms) == 1
            offset = :(1-$(terms[1]))
        else
            offset = :(1)
        end
        push!(iidx_init.args, :($(IIDXs[n])= $offset))
        push!(iidx_inc.args, :($(IIDXs[n]) += 1))
    end

        
    flag_updates = Any[quote end for d=1:D]

    for n=1:N
        if nontrivial_factors[n,D]
            push!(flag_updates[D].args,:(
            $(flags[n,D]) = ($(kdsmmnds[n,D]) in 1:size(C,$D)) && $(kds[D]) in (1+$(jnds[n,D])):(size(C,$D)-$(knds[n,D]));
            ))
        else
            push!(flag_updates[D].args,:($(flags[n,D]) = true))
        end

    end
    for d=1:D-1
        for n=1:N
            if nontrivial_factors[n,d]
                push!(flag_updates[d].args,:(
                $(flags[n,d]) = $(flags[n,d+1]) && ($(kdsmmnds[n,d]) in 1:(size(C,$d))) && $(kds[d]) in (1+$(jnds[n,d])):(size(C,$d)-$(knds[n,d]));
                ))
                
            else
                push!(flag_updates[d].args,:($(flags[n,d]) = $(flags[n,d+1])))
            end
        end
    end
    

    innerloop_stmts = quote
        C[IIDX] = c * C[IIDX]
    end
    

    for n=1:N
        
        tensor_elements = Expr(:call, :(*), :(ts[$n]), Expr(:ref, :B, IIDXs[n]))
        
        for d=1:D
            if nontrivial_factors[n,d]
                push!(tensor_elements.args,:($(diags[n,d])[$(kds[d])-$(jnds[n,d])]))
            end
        end

        push!(innerloop_stmts.args,
            :(
            if $(flags[n,1])
            C[IIDX] += $tensor_elements
#             else
#             println($n, ":    ", $(kds[1]),", ", $(kds[2]))
        end
            )
        )

    end
    
    innerloop = quote
        @simd for $(kds[1])=1:size(C,1)
            $(flag_updates[1])
            @inbounds begin
                $innerloop_stmts
            end
            $iidx_inc
            
        end
    end
    loop = innerloop
    for d=2:D
        loop = quote
            for $(kds[d])=1:size(C,$d)
                $(flag_updates[d])
                $loop
            end
        end
    end
    

    quote
        $diag_assignments
        $iidx_init
        $loop
        C
    end
end
