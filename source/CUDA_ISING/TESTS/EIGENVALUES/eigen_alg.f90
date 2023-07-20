program eigenvalues

   implicit none
   integer :: size
   real, dimension(:, :), allocatable :: A
   real, dimension(:), allocatable :: v, mul

   size = 3
   allocate(A(size, size), v(size))

   A = reshape( (/ &
      -4.0, -5.0, -1.0,     &
      14.0, 13.0 , 0.0 ,     &
      0.0, 0.0, 2.0 &
      /), (/ size, size /))


   call power_method(A, size, 0.000001, 100000)

contains

   subroutine power_method(A, dim, TOL, N_max)
      implicit none
      integer :: N_max, dim, i, p
      real, dimension(dim, dim) :: A
      real, dimension(dim) :: x_0, x
      real :: lambda, TOL, ERR

      ERR = TOL + 1.0
      i = 1
      call random_seed()
      call random_number(x_0)
      p = max_pos(x_0, dim)
      x_0 = x_0 / x_0(p)

      do while ((ERR .ge. TOL) .and. (i .le. N_max))
         x = matrix_vector_mul(A, dim, x_0)
         lambda = x(p)
         p = max_pos(abs_array(x, dim), dim)
         ERR = maxval(abs_array(x_0 - (x / x(p)), dim))
         x_0 = x / x(p)
         i = i + 1
      end do

      if (i .ge. N_max) then
         write(*, *)"The maximum number of iterations exceeded"
      else
         write(*, *)"Max eigenvalue = ", lambda
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

