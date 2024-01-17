program eigenvalues

   implicit none
   integer :: size
   real, dimension(:, :), allocatable :: A, B

   size = 4
   allocate(A(size, size), B(size, size))

   A = reshape( (/ &
      1.0, 234.0, 45.0, 0.2234, &
      234.0, 2.0, -2.45343, -45.0, &
      45.0, -2.45343, 3.0, 23.0, &
      0.2234, -45.0, 23.0, 4.0 &
      /), (/ size, size /))

   
   B = housholder_transformation(A, size)
   
   call QR_method(B, size, 0.000001, 1000)

   deallocate(A, B)
contains

   subroutine QR_method(M, dim, TOL, N_max)
      implicit none
      integer :: dim, N_max, i, j, k, n
      real, dimension(dim, dim) :: M
      real, dimension(dim) :: A, D, X, Z, C, S
      real, dimension(dim - 1) :: B, Y, Q
      real, dimension(dim - 2) :: R
      real :: TOL, SHIFT, lambda, bi, ci, di, mu_1, mu_2, sigma, lambda_1, lambda_2

      ! Fill the arrays A and B with diagonal and off diagonal elements
      do i = 1, dim
         A(i) = M(i, i)
         if (i .ne. dim) then
            B(i) = M(i + 1, i) 
         end if
      end do

      k = 1
      SHIFT = 0.0
      n = dim

      outer: do while (k .le. N_max)
         ! Test for success
         if (abs(B(n - 1)) .le. TOL) then
            lambda = A(n) + SHIFT
            write(*, *) lambda
            n = n - 1
         else
            if (abs(B(1)) .le. TOL) then
               lambda = A(1) + SHIFT
               n = n - 1
               A(1) = A(2)
               do j = 2, n
                  A(j) = A(j + 1)
                  B(j - 1) = B(j)
               end do
            end if
         end if
         if (n .eq. 0) then
            exit
         end if
         if (n .eq. 1) then
            lambda = A(1) + SHIFT
            write(*, *)lambda
            exit
         end if
         do j = 3, n - 1
            if (abs(B(j - 1)) .le. TOL) then
               write(*, *)"Split into", A(1:j - 1), ", ", B(1:j - 2)
               write(*, *)"and"
               write(*, *)A(j:n), ", ", B(j:n - 1)
               write(*, *)"with SHIFT = ", SHIFT
               exit outer
            end if
         end do

         ! Compute shift
         bi = - (A(n - 1) + A(n))
         ci = a(n) * a(n - 1) - (B(n - 1)) ** 2
         di = sqrt(bi ** 2 - 4 * ci)
         if (bi .gt. 0.0) then
            mu_1 = - (2.0 * ci) / (bi + di)
            mu_2 = - (bi + di) / 2.0
         else
            mu_2 =  (2.0 * ci) / (di - bi)
            mu_1 =  (di - bi) / 2.0
         end if
         if (n .eq. 2) then
            lambda_1 = mu_1 + SHIFT
            lambda_2 = mu_2 + SHIFT
            write(*, *) lambda_1
            write(*, *) lambda_2
            exit
         end if
         if (abs(mu_1 - A(n)) .le. abs(mu_2 - A(n))) then
            sigma = mu_1
         else
            sigma = mu_2
         end if

         ! Accumulate the shift
         SHIFT = SHIFT + sigma

         ! Perform the shift
         do j = 1, n
            D(j) = A(j) - sigma
         end do

         ! Compute R^(K)
         X(1) = D(1)
         Y(1) = B(1)
         do j = 2, n
            Z(j - 1) = sqrt(X(j - 1) ** 2 + (B(j - 1)) ** 2)
            C(j) = X(j - 1) / Z(j - 1)
            S(j) = B(j - 1) / Z(j - 1)
            Q(j - 1) = C(j) * Y(j - 1) + S(j) * D(j)
            X(j) = - S(j) * Y(j - 1) + C(j) * D(j)
            if (j .ne. n) then
               R(j - 1) = S(j) * B(j)
               Y(j) = C(j) * B(j)
            end if
         end do

         ! Compute A^(k + 1)
         Z(n) = X(n)
         A(1) = S(2) * Q(1) + C(2) * Z(1)
         B(1) = S(2) * Z(2)
         do j= 2, (n - 1)
            A(j) = S(j + 1) * Q(j) + C(j) * C(j + 1) * Z(j)
            B(j) = S(j + 1) * Z(j + 1)
         end do
         A(n) = C(n) * Z(n)
         k = k + 1
      end do outer
      
      if (k .gt. N_max) then
         write(*, *)"Maximum number of iterations exceeded"
      end if

   end subroutine QR_method

   function housholder_transformation(A, dim)
      implicit none
      integer :: dim, k, j
      real, dimension(dim, dim) :: A, P, housholder_transformation, id
      real, dimension(dim) :: w
      real, dimension(:), allocatable :: a_slice
      real :: r, alpha

      id = 0.0
      do j = 1, dim
         id(j, j) = 1.0
      end do

      do k = 1, (dim - 2)
         allocate(a_slice(dim - k))
         a_slice = A(k + 1:, k)
         alpha = - sign(1.0, A(k + 1, k)) * norm2(a_slice)
         r = sqrt(0.5 *  alpha * (alpha - A(k + 1, k)))
         w(:k) = 0.0
         w(k + 1) = (A(k + 1, k) - alpha) / (2.0 * r)
         do j = k + 2, dim
            w(j) = A(j, k) / (2.0 * r)
         end do
         P = id - 2.0 * vector_vectort_mul(w, w, dim)
         A = matmul(matmul(P, A), P)
         if ((k + 2) .le. (dim)) then
            A(k + 2:, k) = 0.0
            A(k, k + 2:) = 0.0
         end if 
         deallocate(a_slice)
      end do

      housholder_transformation = A

   end function housholder_transformation


   function vector_vectort_mul(v1, vt2, dim)
      implicit none
      integer :: dim, i, j
      real, dimension(dim) :: v1, vt2
      real, dimension(dim, dim) :: vector_vectort_mul

      do i = 1, dim
         do j = 1, dim
            vector_vectort_mul(i, j) = v1(i) * vt2(j)
         end do
      end do

   end function vector_vectort_mul

   subroutine symmetric_power_method(A, dim, TOL, N_max)
      integer :: N_max, dim, i
      real, dimension(dim, dim) :: A
      real, dimension(dim) :: x_0, x
      real :: lambda, TOL, ERR

      ERR = TOL + 1.0
      i = 1
      call random_seed()
      call random_number(x_0)

      x_0 = x_0 / NORM2(x_0)

      do while ((i .le. N_max) .and. (ERR .ge. TOL))
         x = matrix_vector_mul(A, dim, x_0)
         lambda = DOT_PRODUCT(x_0, x)
         ERR = NORM2(x_0 - (x / NORM2(x)))
         x_0 = x / NORM2(x)
         i = i + 1
      end do

      if (i .ge. N_max) then
         write(*, *)"The maximum number of iterations exceeded"
         write(*, *)"Max eigenvalue = ", lambda
      else
         write(*, *)"Max eigenvalue = ", lambda
         write(*, *)"Iteration = ", i
      end if


   end subroutine symmetric_power_method

   subroutine power_method(A, dim, TOL, N_max)
      implicit none
      integer :: N_max, dim, i, p
      real, dimension(dim, dim) :: A
      real, dimension(dim) :: x_0, x
      real :: lambda, lambda_0, lambda_1, lambda_AC, TOL, ERR

      ERR = TOL + 1.0
      i = 1
      call random_seed()
      call random_number(x_0)
      p = max_pos(x_0, dim)
      x_0 = x_0 / x_0(p)
      lambda_0 = 0.0
      lambda_1 = 0.0

      do while ((ERR .ge. TOL) .and. (i .le. N_max))
         x = matrix_vector_mul(A, dim, x_0)
         lambda = x(p)
         lambda_AC = lambda_0 - ((lambda_1 - lambda_0) ** 2) / (lambda - 2.0 * lambda_1 + lambda_0)
         p = max_pos(abs_array(x, dim), dim)
         ERR = maxval(abs_array(x_0 - (x / x(p)), dim))
         x_0 = x / x(p)
         lambda_0 = lambda_1
         lambda_1 = lambda
         i = i + 1
      end do

      if (i .ge. N_max) then
         write(*, *)"The maximum number of iterations exceeded"
         write(*, *)"Max eigenvalue = ", lambda
         write(*, *)"Max eigenvalue (AC) = ", lambda_AC
      else
         if (i .ge. 4) then
            write(*, *)"Max eigenvalue (AC) = ", lambda_AC
            write(*, *)"Max eigenvalue = ", lambda
            write(*, *)"Iteration = ", i
         else
            write(*, *)"Max eigenvalue = ", lambda
         end if
      end if

   end subroutine power_method

   function abs_array(array, dim)
      implicit none
      integer :: i, dim
      real, dimension(dim) :: array, abs_array

      do i = 1, dim
         abs_array(i) = abs(array(i))
      end do

   end function abs_array

   function max_pos(array, dim)
      implicit none
      integer :: i, dim, max_pos
      real, dimension(dim) :: array
      real :: max, try

      max = array(1)
      max_pos = 1
      do i = 1, dim
         try = array(1)
         if (max .le. try) then
            max = try
            max_pos = i
         end if
      end do

   end function max_pos


   function matrix_vector_mul(A, dim, v)

      implicit none
      integer :: dim, i, j
      real, dimension(dim) :: v
      real, dimension(dim, dim) :: A
      real, dimension(dim) :: matrix_vector_mul

      do i = 1, dim
         matrix_vector_mul(i) = 0.0
         do j = 1, dim
            matrix_vector_mul(i) = matrix_vector_mul(i) + A(i, j) * v(j)
         end do
      end do

   end function matrix_vector_mul

end program eigenvalues

