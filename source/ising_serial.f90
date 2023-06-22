program Ising_Monte_Carlo_Serial
    implicit none
    integer :: N_size, N_term, N_misures, N_decorr
    integer :: Start,seed, i_mis, i_decor, i_count, i_term
    integer, dimension(:,:), allocatable :: Field
    integer, parameter :: dp = kind(1.0d0)
    integer, dimension(:), allocatable :: aft,bfr
    real :: Beta, B
    real(kind=dp) :: mgn, eng
    character(len=3) :: name

    seed = -145
    N_term = 100000
    N_misures = 2**20
    N_size = 40
    N_decorr = 1
    Beta = 0.35
    B = 0.0
    Start = 1

    allocate(Field(N_size, N_size))
    allocate(aft(N_size))
    allocate(bfr(N_size))
    call geometry(N_size)

    do i_count = 1, 100

        write(name, 2) i_count
        2 format(I3)

        open(unit=i_count, action='write', file=trim(name)//'.txt')

        write(i_count, '(A4,I3)') '# L ', N_size
        write(i_count, '(A4,F5.3)') '# B ', Beta

        call init_field(N_size, Field, Start)

        do i_term = 1, N_term
            call update_metro(N_size, Beta, Field, B)
        end do

        do i_mis = 1, N_misures
            do i_decor = 1, N_decorr
                call update_metro(N_size, Beta, Field, B)
            end do 
            mgn = Magn(N_size, Field)
            write(i_count, 1) mgn
            1 format(f21.18)
        end do
        close(i_count)

        Beta = Beta + 0.002

    end do

    deallocate(Field, aft, bfr)

    contains

    subroutine geometry(sz)
        implicit none
        integer:: sz, i
        do i = 1, sz - 1
            aft(i) = i + 1
            bfr(i + 1) = i
        end do
        aft(sz) = 1
        bfr(1) = sz
    end subroutine geometry

    subroutine init_field(sz,fld,st)
        implicit none
        integer :: sz, st, i, j
        integer, dimension(sz, sz) :: fld
        real :: rnd
        if (st .eq. 0) then
            do i = 1, sz
                do j = 1, sz
                    fld(i, j) = 1
                end do
            end do
        else
            if (st .eq. 1) then
                do i = 1, sz
                    do j = 1, sz
                        rnd = rand1(seed)
                        if (rnd .gt. 0.5) then
                            fld(i, j) = 1
                        else
                            fld(i, j) = -1
                        end if
                    end do
                end do
            end if 
        end if
    end subroutine init_field

    function Energy(sz, fld, h)
        implicit none
        real(kind=dp) :: Energy
        real :: h
        integer :: i, j, sz, force
        integer, dimension(sz,sz) :: fld
        Energy = 0.0
        do i = 1, sz
            do j = 1, sz
                force = fld(i, bfr(j)) + fld(i, aft(j)) + fld(bfr(i), j) + fld(aft(i), j)
                Energy = Energy - real(fld(i,j), dp) * (0.5_dp * real(force, dp) + real(h, dp))
            end do
        end do
        Energy = Energy / (real(sz**2, dp))
    end function Energy

    function Magn(sz, fld)
        implicit none
        real(kind=dp):: Magn
        integer :: i, j, sz
        integer, dimension(sz, sz):: fld
        Magn = 0.0
        do i = 1, sz
            do j = 1, sz
                Magn = Magn + real(fld(i,j), dp)
            end do
        end do
        Magn = abs(Magn) / real(sz**2, dp)
    end function Magn

    subroutine update_metro(sz, bt, fld, h)
        implicit none
        integer :: i_try, sz, x, y, force, spin
        integer, dimension(sz, sz) :: fld
        real :: bt, h, expo, rate, rdm
        do i_try = 1, sz**2
            x = floor(rand1(seed) * sz) + 1
            y = floor(rand1(seed) * sz) + 1
            spin = fld(x, y)
            force = fld(x, bfr(y)) + fld(x, aft(y)) + fld(bfr(x), y) + fld(aft(x), y)
            expo = -2.0 * (real(spin)) * bt * (real(force) + h)
            if (expo .ge. 0.0) then
                fld(x, y) = - spin
            else
                rate = exp(expo)
                rdm = rand1(seed)
                if (rdm .le. rate) then
                    fld(x,y) = - spin
                end if
            end if
        end do
    end subroutine update_metro

    function rand1(idum)
        IMPLICIT NONE
        INTEGER, PARAMETER :: K4B = selected_int_kind(9)
        INTEGER(K4B), INTENT(INOUT) :: idum
        REAL :: rand1
        INTEGER(K4B), PARAMETER :: IA = 16807, IM = 2147483647, IQ = 127773, IR = 2836
        REAL, SAVE :: am
        INTEGER(K4B), SAVE :: ix = -1, iy = -1, k
        if (idum <= 0 .or. iy < 0) then
            am = nearest(1.0, -1.0) / IM
            iy = ior(ieor(888889999, abs(idum)), 1)
            ix = ieor(777755555, abs(idum))
            idum = abs(idum) + 1
        end if
        ix = ieor(ix, ishft(ix, 13)) 
        ix = ieor(ix, ishft(ix, -17))
        ix = ieor(ix, ishft(ix, 5))
        k = iy / IQ 
        iy = IA * (iy - k * IQ) - IR * k
        if (iy < 0) then 
            iy = iy + IM
        end if
        rand1 = am * ior(iand(IM, ieor(ix, iy)), 1)
    end function rand1

end program Ising_Monte_Carlo_Serial