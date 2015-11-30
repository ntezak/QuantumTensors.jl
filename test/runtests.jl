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

function test_kron()
    N1 = 5
    N2 = 10
    tol = 1e-10
    a1 = destroy(N1)
    a2 = destroy(N2)
    @test norm((kron(a1,a2) |> full) - kron(a1|>full, a2|> full)) < tol
end

function test_A_mul_B(N1=10,N2=10)
    tol = 1e-5
    a1 = destroy(N1)
    ad2 = create(N2)
    a1ad2 = tensor(a1,ad2)
    p1 = randn(N1)
    p2 = randn(N2)
    x = tensor(p1, p2)
    println("Tensor prod:")
    y1 = zeros(N1,N2)
    A_mul_B!(y1, a1ad2, x)

    @time begin 
        y1 = zeros(N1,N2)
        A_mul_B!(y1, a1ad2, x)
    end
    
    println("Tensor prod (inplace):")
    y5 = zeros(N1,N2)
    @time begin 
        A_mul_B!(y5, a1ad2, x)
    end
    @test norm(y1[:] - y5[:]) < tol


    println("Tensor prod kernel:")
    y6 = zeros(N1,N2)
    k! = make_mul_kernel(a1ad2)
    k!(y6, x)
    @time begin 
        k!(y6, x)
    end
    @test norm(y1[:] - y6[:]) < tol



    println("Tensor prod with addition:")
    y4 = zeros(N1,N2)
    A_mul_B!(2., a1ad2, x, 3., y4)

    @time begin 
        y4 = zeros(N1,N2)
        A_mul_B!(2., a1ad2, x, 3., y4)
    end
    @test norm(2*y1[:] - y4[:]) < tol


    println(methods(TensorOpProductKernel))
    println("TensorOpProductKernel:")
    y10 = zeros(N1,N2)
    A_mul_B_generated!(2., a1ad2, x, 3., y10)
    A_mul_B_generated!(2., a1ad2.kernel, x, 3., y10)
    y10 = zeros(N1,N2)
    gc()
    @time begin     
        A_mul_B_generated!(2., a1ad2.kernel, x, 3., y10)
    end
    @test norm(2*y1[:] - y10[:]) < tol

    

    println("Tensor prod with addition (inplace):")
    gc()
    @time begin 
        A_mul_B!(2., a1ad2, x, 3., y4)
    end
    @test norm(8*y1[:] - y4[:]) < tol

    println("Tensor prod kernel with addition:")
    y7 = zeros(N1,N2)
    k! = make_plus_mul_kernel(Float64,a1ad2)
    k!(2., x, 3., y7)
    gc()
    @time begin 
        k!(2., x, 3., y7)
    end
    @test norm(8*y1[:] - y7[:]) < tol




    # function kernel!{T}(a::T, Av1::Vector{T}, Av2::Vector{T}, sB::SubArray{T,2}, sC::SubArray{T,2})
    #     for k2 = 1:size(sC,2)
    #         @simd for k1 = 1:size(sC,1)
    #             @inbounds sC[k1,k2] = sC[k1,k2] + a * Av1[k1] * Av2[k2] * sB[k1,k2]
    #         end
    #     end
    # end

    function kernel1!(a, Av1, Av2, sB, sC)
        for k2 = 1:size(sC,2)
            @simd for k1 = 1:size(sC,1)
                @inbounds sC[k1,k2] = sC[k1,k2] + a * Av1[k1] * Av2[k2] * sB[k1,k2]
            end
        end
    end


    function custom_A_mul_B2_1!{T}(a::T, A, B::Array{T,2}, c::T, C::Array{T,2})
        size(B) == size(C) || error("Incompatible sizes")
        if c != one(T)
            if c == zero(T)
                fill!(C,c)
            else 
                scale!(C,c)
            end 
        end 
        iidx1, iidx2 = iidx(A)
        oidx1, oidx2 = oidx(A)

        Av1 = (A.ops[1].vals)::Array{T,1}
        Av2 = (A.ops[2].vals)::Array{T,1}

        sB = sub(B, iidx1, iidx2)
        sC = sub(C, oidx1, oidx2)

        kernel1!(a, Av1, Av2, sB, sC)
        C
    end


    println("Custom function 1:")
    y8 = zeros(N1,N2)
    custom_A_mul_B2_1!(2., a1ad2, x, 3., y8)
    gc()
    @time begin 
        custom_A_mul_B2_1!(2., a1ad2, x, 3., y8)
    end
    @test norm(8*y1[:] - y8[:]) < tol

    y8 = 0
    gc()

    function kernel2!(a, Av1, Av2, sB, c, sC)
        for k2 = 1:size(sC,2)
            @simd for k1 = 1:size(sC,1)
                @inbounds sC[k1,k2] = c * sC[k1,k2] + a * Av1[k1] * Av2[k2] * sB[k1,k2]
            end
        end
    end


    function custom_A_mul_B2_2!{T}(a::T, A, B::Array{T,2}, c::T, C::Array{T,2})
        size(B) == size(C) || error("Incompatible sizes")
        iidx1, iidx2 = iidx(A)
        oidx1, oidx2 = oidx(A)

        Av1 = (A.ops[1].vals)::Array{T,1}
        Av2 = (A.ops[2].vals)::Array{T,1}

        sB = sub(B, iidx1, iidx2)
        sC = sub(C, oidx1, oidx2)

        kernel2!(a, Av1, Av2, sB, c, sC)
        C
    end


    println("Custom function 2:")
    y8 = zeros(N1,N2)
    custom_A_mul_B2_2!(2., a1ad2, x, 3., y8)
    gc()
    @time begin 
        custom_A_mul_B2_2!(2., a1ad2, x, 3., y8)
    end
    @test norm(8*y1[:] - y8[:]) < tol

    y8 = 0
    gc()

    # function kernel_p!(a, Av1, Av2, sB, sC)
    #     @sync @parallel for k2 = 1:size(sC,2)
    #         @simd for k1 = 1:size(sC,1)
    #             @inbounds sC[k1,k2] = sC[k1,k2] + a * Av1[k1] * Av2[k2] * sB[k1,k2]
    #         end
    #     end
    # end


    # function custom_A_mul_B2_p!{T}(a::T, A, B::SharedArray{T,2}, c::T, C::SharedArray{T,2})
    #     size(B) == size(C) || error("Incompatible sizes")
    #     if c != one(T)
    #         if c == zero(T)
    #             fill!(C,c)
    #         else 
    #             scale!(C,c)
    #         end 
    #     end 
    #     iidx1, iidx2 = iidx(A)
    #     oidx1, oidx2 = oidx(A)

    #     Av1 = (A.ops[1].vals)::SharedArray{T,1}
    #     Av2 = (A.ops[2].vals)::SharedArray{T,1}

    #     sB = sub(B, iidx1, iidx2)
    #     sC = sub(C, oidx1, oidx2)

    #     kernel_p!(a, Av1, Av2, sB, sC)
    #     C
    # end


    # println("Custom function parallel:")
    # y9 = SharedArray(Float64, N1, N2)
    # @test norm(y9) == 0
    # xp = share(x)
    # @test norm(xp[:] - x[:]) == 0
    # a1ad2p = share(a1ad2)
    # custom_A_mul_B2_p!(2., a1ad2p, xp, 3., y9)
    # @time begin 
    #     custom_A_mul_B2_p!(2., a1ad2p, xp, 3., y9)
    # end
    # @test norm(8*y1[:] - y9[:]) < tol

    # xp = 0
    # y9 = 0
    # gc()

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
    gc()
    @time A_mul_B!(2., a1ad2sp, xv, 1., y3)
    @test norm(3*y1[:] - y3[:]) < tol


    println("Size of vectors: ", Base.summarysize(x)/1024^2, " MB")
    println("Size of sparse matrix rep: ", Base.summarysize(a1ad2sp)/1024^2, " MB")
    println("Size of tensor matrix rep: ", Base.summarysize(a1ad2)/1024^2, " MB")


    



end

addprocs(3)
versioninfo()
test_to_sparse()
test_to_full()
test_transpose()
test_mul()
test_kron()
test_A_mul_B(10000,5000)


