python_sources = [
  '__init__.py',
  'consistent_initial_conditions.py',
  'dae_problem.py',
  'benchmark.py',
]

py.install_sources(
  python_sources,
  subdir: 'dae4py'
)

subdir('bdf')
subdir('butcher_tableau')
subdir('erk')
subdir('irk')
subdir('math')
subdir('radau')

#if enable_fortran
#  if blas_dep.found() and lapack_dep.found()
#    message('Add fortran wrapper since all dependencies are found')
#    subdir('fortran')
#  else
#    message('Skipp fortran wrapper since not dependencies are found')
#  endif
#endif