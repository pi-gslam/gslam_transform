######################################################################
# Automatically generated by qmake (2.01a) ?? 10? 22 18:23:54 2014
######################################################################

TEMPLATE = app
TARGET = 
DEPENDPATH += . benchmark doc functions internal optimization regressions test
INCLUDEPATH += . benchmark regressions ..

# Input
HEADERS += Cholesky.h \
           determinant.h \
           gauss_jordan.h \
           gaussian_elimination.h \
           generated.h \
           GR_SVD.h \
           helpers.h \
           irls.h \
           lapack.h \
           Lapack_Cholesky.h \
           LU.h \
           QR.h \
           QR_Lapack.h \
           se2.h \
           se3.h \
           sim2.h \
           sim3.h \
           sl.h \
           so2.h \
           so3.h \
           SVD.h \
           SymEigen.h \
           TooN.h \
           wls.h \
           doc/cramer.h \
           doc/documentation.h \
           doc/linoperatorsdoc.h \
           functions/derivatives.h \
           functions/fadbad.h \
           internal/allocator.hh \
           internal/builtin_typeof.h \
           internal/comma.hh \
           internal/config.hh \
           internal/data.hh \
           internal/data_functions.hh \
           internal/dchecktest.hh \
           internal/debug.hh \
           internal/deprecated.hh \
           internal/diagmatrix.h \
           internal/introspection.hh \
           internal/make_vector.hh \
           internal/matrix.hh \
           internal/mbase.hh \
           internal/objects.h \
           internal/operators.hh \
           internal/overfill_error.hh \
           internal/planar_complex.hh \
           internal/reference.hh \
           internal/size_mismatch.hh \
           internal/slice_error.hh \
           internal/typeof.hh \
           internal/vbase.hh \
           internal/vector.hh \
           optimization/brent.h \
           optimization/conjugate_gradient.h \
           optimization/downhill_simplex.h \
           optimization/golden_section.h \
           regressions/regression.h \
           benchmark/solvers.cc
SOURCES += benchmark/solve_ax_equals_b.cc \
           benchmark/solvers.cc \
           regressions/chol_lapack.cc \
           regressions/chol_toon.cc \
           regressions/complex.cc \
           regressions/determinant.cc \
           regressions/diagonal_matrix.cc \
           regressions/eigen-sqrt.cc \
           regressions/fill.cc \
           regressions/gauss_jordan.cc \
           regressions/gaussian_elimination.cc \
           regressions/gr_svd.cc \
           regressions/lu.cc \
           regressions/qr.cc \
           regressions/simplex.cc \
           regressions/slice.cc \
           regressions/so3.cc \
           regressions/sym_eigen.cc \
           regressions/vector_resize.cc \
           regressions/zeros.cc \
           test/as_foo.cc \
           test/brent_test.cc \
           test/cg_test.cc \
           test/cramer.cc \
           test/deriv_test.cc \
           test/diagslice.cc \
           test/dynamic_test.cc \
           test/fadbad.cpp \
           test/gaussian_elimination_test.cc \
           test/golden_test.cc \
           test/identity_test.cc \
           test/log.cc \
           test/lutest.cc \
           test/make_vector.cc \
           test/makevector.cc \
           test/mat_test.cc \
           test/mat_test2.cc \
           test/mmult_test.cc \
           test/normalize_test.cc \
           test/normalize_test2.cc \
           test/scalars.cc \
           test/sl.cc \
           test/slice_test.cc \
           test/svd_test.cc \
           test/SXX_test.cc \
           test/sym.cc \
           test/test2.cc \
           test/test3.cc \
           test/test_data.cc \
           test/test_foreign.cc \
           test/un_project.cc \
           test/vec_test.cc
