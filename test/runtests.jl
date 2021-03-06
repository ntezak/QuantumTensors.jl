using QuantumTensors
using Base.Test



function test_to_sparse()
    tol = 1e-10
    N = 5
    @test norm(((destroy(N) |> sparse) - spdiagm(sqrt(1:(N-1)),1,N,N))|>full )< tol
    @test norm(((create(N) |> sparse) - spdiagm(sqrt(1:(N-1)),-1,N,N))|>full )< tol
    @test norm(((jz(N) |> sparse) - spdiagm(-N:N,0,2N+1,2N+1))|>full) < tol
    
    jzm = jz(N) |> sparse
    jp = jplus(N) |> sparse
    jm1 = jp'
    jm2 = jminus(N) |> sparse
    @test norm((jm1-jm2)|> full) < tol
    
    comm_rel = jp*jm1 - jm1*jp
    @test norm((comm_rel-2jzm)|>full) < tol
end

function test_to_full()
    tol = 1e-10
    N = 5
    @test norm((destroy(N) |> full) - diagm(sqrt(1:(N-1)),1))< tol
    @test norm((create(N) |> full) - diagm(sqrt(1:(N-1)),-1))< tol
    @test norm((sigma(2,1,2) |> full) - [0 1; 0 0])< tol
    @test norm((sigma(2,1,1) |> full) - [1 0; 0 0])< tol
    @test norm((sigma(2,2,1) |> full) - [0 0; 1 0])< tol
    @test norm((sigma(2,2,2) |> full) - [0 0; 0 1])< tol
end

function test_transpose()
    tol = 1e-10
    @test norm((sigma(2,1,2) |>full) - (sigma(2,2,1) |> transpose |>full)) < tol
    @test norm((destroy(5) |>full) - (create(5) |> transpose |>full)) < tol
end


function test_convert_promote_ops()
  tol = 1e-10
  N = 5
  # a = destroy(N)
  # ac = convert(DiagonalOp{Complex128}, a)
  idt = IdentityOp{N}()
  idtf = convert(DiagonalOp{Float64}, idt)
  idtz1 = convert(DiagonalOp{Complex128}, idt)
  idtz2 = convert(DiagonalOp{Complex128}, idtf)
  idtz3, idtz4 = promote(idtf, idtz1)
  @test idtz4 === idtz1
  @test isa(idtf, DiagonalOp{Float64})
  @test isa(idtz1, DiagonalOp{Complex128})
  @test isa(idtz2, DiagonalOp{Complex128})
  @test isa(idtz3, DiagonalOp{Complex128})
  @test idtf.jkm == (0,0,0)
  @test idtz1.jkm == (0,0,0)
  @test idtz2.jkm == (0,0,0)
  @test idtz3.jkm == (0,0,0)
  @test norm(idtf.vals - ones(N)) < tol
  @test norm(idtz1.vals - ones(N)) < tol
  @test norm(idtz2.vals - ones(N)) < tol
  @test norm(idtz3.vals - ones(N)) < tol
  
  @test all_close(idt, idtf)
  @test all_close(idt, idtz1)
  @test all_close(idt, idtz2)
  @test all_close(idt, idtz3)
  @test all_close(idt, idtz4)
  @test all_close(idtf, idtz1)
  @test all_close(idtf, idtz2)
  @test all_close(idtf, idtz3)
  @test all_close(idtf, idtz4)
end


function test_convert_promote_tops()
  tol = 1e-10
  N = 5
  a = destroy(N)
  bc = convert(DiagonalOp{Complex128}, destroy(N))
  
  ac, bc = promote(a, bc)
  @test bc === bc
  @test all_close(ac, a)
  abc = tensor(a, bc)
  acbc = tensor(ac, bc)
  
  @test typeof(abc) === typeof(acbc)
  @test all_close(abc, acbc)
end

function test_convert_promote_topsum()
  tol = 1e-10
  N = 5
  a = destroy(N)
  bc = convert(DiagonalOp{Complex128}, destroy(N))
  ii = IdentityOp{N}()
  
  ac, bc = promote(a, bc)
  @test bc === bc
  @test all_close(ac, a)
  abc = tensor(a, bc)
  acbc = tensor(ac, bc)
  
  iibc = tensor(ii, bc)
  aii = tensor(a, ii)
  
  tops = TensorOpProductSum(abc, iibc, aii)
end




function test_mul()
    tol = 1e-10
    a = destroy(5)
    @test norm((a*a |> full) -(a|> full)*(a|> full)) < tol
    @test norm((a*a' |> full) -(a|> full)*(a|> full)')<tol
end



function test_apply_serial(N=20,tol=1e-8)
  println("Testing serial application of operators")
  a = destroy(N)
  b = destroy(N)
  c = destroy(N)
  II = IdentityOp{N}()
  tops = (tensor(a', b, II), tensor(a, b', II), tensor(II, II, c'*c))
  topsum = TensorOpProductSum(tops...)

  B = randn(N,N,N)
  C = zeros(N,N,N)
  coeffs = [1.,1.,1.]
  apply_serial!(coeffs, topsum, B, C)
  # Profile.clear()
  println("apply_serial!: TensorOpProductSum ")
  gc()
  @time for k=1:10
      apply_serial!(coeffs, topsum, B,C)
  end
  

  topskarr = collect([TensorOpProductKernel(top) for top in tops]);
  topsarr = collect(tops);

  println("apply_serial!: [TensorOpProductKernel]")
  Cpp = zeros(C)
  apply_serial!(coeffs, topskarr, B, Cpp)
  gc()
  @time for k=1:10
      apply_serial!(coeffs, topskarr, B, Cpp)
  end
  
  @test (norm((Cpp-C)[:])) < tol

  println("apply_serial!: [TensorOpProduct]")
  Cpp = zeros(C)
  apply_serial!(coeffs, topsarr, B, Cpp)
  gc()
  @time for k=1:10
      apply_serial!(coeffs, topsarr, B, Cpp)
  end
  @test norm((Cpp-C)[:]) < tol

  Cp = zeros(C)
  Cpf = sub(Cp,1:length(C));
  Bf = sub(B,1:length(B));
  top_sps = [sparse(top) for top = tops]
  combinedsp = sum(top_sps)
  println("combined CSR matrix")
  A_mul_B!(Cpf, combinedsp, Bf)
  gc()
  @time for k=1:10
      A_mul_B!(Cpf, combinedsp, Bf)
  end
  @test norm((Cp-C)[:]) < tol
  
  
  println("apply_serial!: [CSR-mat]")
  apply_serial!(coeffs, top_sps, Bf, Cpf)
  gc()
  @time for k=1:10
      apply_serial!(coeffs, top_sps, Bf, Cpf)
  end
  @test norm((Cp-C)[:]) < tol
end

module SymbolicTest
  const sqrt_arr1 = sqrt(1:1000000)
  const sqrt_arr2 = sqrt(1:1000000)
  function a1ad2_symbolic!(y, x, α=0, β=1)
    N1, N2=size(x)
    
    if α != 1
      if α == 0
        fill!(y, 0)
      else
        scale!(y, α)
      end
    end
      
    
    for l=1:(N2-1)
      @simd for k=1:(N1-1)
        @inbounds y[k,l+1] +=  β * sqrt_arr1[k]*sqrt_arr2[l]*x[k+1,l]
      end
    end
    y
  end
  function a1ad2_symbolic2!(y, x, α=0, β=1)
    N1, N2=size(x)
    
    if α != 1
      if α == 0
        fill!(y, 0)
      else
        scale!(y, α)
      end
    end
    
    offset1 = -1
    offset2 = 1
    
    oIDX = 1
    iIDX = 1 - stride(x,1)*offset1 - stride(x,2)*offset2
    for l=1:N2
      lflag = 2 <= l <=N2
      @simd for k=1:N1
        kflag = 1 <= k <= N1-1
        if kflag && lflag
          @inbounds y[oIDX] +=  β * sqrt_arr1[k]*sqrt_arr2[l-1]*x[iIDX]
        end
        oIDX +=1
        iIDX +=1
      end
    end
    y
  end
  function a1ad2_symbolic3!(y, x, α=0, β=1)
    N1, N2=size(x)
    
    if α != 1
      if α == 0
        fill!(y, 0)
      else
        scale!(y, α)
      end
    end
    
    offset1 = -1
    offset2 = 1
    
    oIDX = 1
    iIDX = 1 - stride(x,1)*offset1 - stride(x,2)*offset2
    for l=1:N2
      lflag = 2 <= l <=N2
      for k=1:N1
        kflag = 1 <= k <= N1-1
        if kflag && lflag
          @inbounds y[oIDX] +=  β * sqrt_arr1[k]*sqrt_arr2[l-1]*x[iIDX]
        end
        oIDX +=1
        iIDX +=1
      end
    end
    y
  end
end

function test_A_mul_B(N1=10,N2=10)
    println("Testing application of single tensor product operators to vectors")
    tol = 1e-5
    a1 = destroy(N1)
    ad2 = create(N2)
    a1ad2 = tensor(a1,ad2)
    p1 = randn(N1)
    p2 = randn(N2)
    x = tensor(p1, p2)



    println("Tensor prod with allocation:")
    y1 = zeros(N1,N2)
    A_mul_B!(y1, a1ad2, x)
    gc()
    @time begin
        y1 = A_mul_B!(zeros(N1,N2), a1ad2, x)
    end
    
    println("Tensor prod (inplace):")
    y5 = zeros(N1,N2)
    gc()
    @time begin 
        A_mul_B!(y5, a1ad2, x)
    end
    @test norm(y1[:] - y5[:]) < tol

    println("Direct usage of TensorOpProductKernel:")
    y10 = zeros(N1,N2)
    A_mul_B!(2., a1ad2.kernel, x, 3., y10)
    y10 = zeros(N1,N2)
    gc()
    @time begin     
        A_mul_B!(2., a1ad2.kernel, x, 3., y10)
    end
    @test norm(2*y1[:] - y10[:]) < tol

    println("'Symbolic' application of specialized function:")
    y11 = zeros(N1,N2)
    SymbolicTest.a1ad2_symbolic!(y11, x, 3., 2.)
    y11 *= 0
    gc()
    @time begin     
        SymbolicTest.a1ad2_symbolic!(y11, x, 3., 2.)
    end
    @test norm(2*y1[:] - y11[:]) < tol

    println("'Symbolic' application of second specialized function:")
    y12 = zeros(N1,N2)
    SymbolicTest.a1ad2_symbolic2!(y12, x, 3., 2.)
    y12 *= 0
    gc()
    @time begin     
        SymbolicTest.a1ad2_symbolic2!(y12, x, 3., 2.)
    end
    @test norm(2*y1[:] - y12[:]) < tol

    println("'Symbolic' application of third specialized function:")
    y12 = zeros(N1,N2)
    SymbolicTest.a1ad2_symbolic3!(y12, x, 3., 2.)
    y12 *= 0
    gc()
    @time begin     
        SymbolicTest.a1ad2_symbolic3!(y12, x, 3., 2.)
    end
    @test norm(2*y1[:] - y12[:]) < tol

    println("Tensorproduct of individual products:")
    y2 = tensor(((a1|>sparse)*p1),((ad2|>sparse)*p2))
    gc()
    @time y2 = tensor(((a1|>sparse)*p1),((ad2|>sparse)*p2))
    @test norm(y1[:] - y2[:]) < tol
    
    println("CSR mat-vec:")
    
    xv = x[:]

    a1ad2sp = (a1ad2 |> sparse)
    y3 = a1ad2sp * xv
    gc()
    @time y3 = a1ad2sp * xv
    @test norm(y1[:] - y3[:]) < tol

    println("CSR mat-vec (inplace):")
    A_mul_B!(2., a1ad2sp, xv, 1., y3)
    y3[:] = y1[:]
    gc()
    @time A_mul_B!(2., a1ad2sp, xv, 1., y3)
    @test norm(3*y1[:] - y3[:]) < tol


    println("Size of vectors: ", Base.summarysize(x)/1024^2, " MB")
    println("Size of sparse matrix rep: ", Base.summarysize(a1ad2sp)/1024^2, " MB")
    println("Size of tensor matrix rep: ", Base.summarysize(a1ad2)/1024^2, " MB")
end

# addprocs(3)
versioninfo()
test_to_sparse()
test_to_full()
test_transpose()
test_mul()
test_convert_promote_ops()
test_convert_promote_tops()
test_convert_promote_topsum()
test_apply_serial(20)
test_A_mul_B(100,100)
