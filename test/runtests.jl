using QuantumTensors
using Base.Test



function test_to_sparse()
    tol = 1e-10
    N = 5
    @test norm(((destroy(N) |> sparse) - spdiagm(sqrt(1:(N-1)),1,N,N))|>full )< tol
    @test norm(((create(N) |> sparse) - spdiagm(sqrt(1:(N-1)),-1,N,N))|>full )< tol
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

function test_mul()
    tol = 1e-10
    a = destroy(5)
    @test norm((a*a |> full) -(a|> full)*(a|> full)) < tol
    @test norm((a*a' |> full) -(a|> full)*(a|> full)')<tol
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
test_A_mul_B(1000,1000)
