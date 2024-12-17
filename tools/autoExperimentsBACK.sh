#!/bin/sh

MATRIX_FOLDER=/home/paul/UNSA/directs/spike_pstrsv/exs
echo "******************* CASE-1: ORIGINAL ******************"
for matrix in ${MATRIX_FOLDER}/Original/Symmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		./runtimeAnalysis $matrix $nthreads U Y N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

for matrix in ${MATRIX_FOLDER}/Original/PatternSymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		./runtimeAnalysis $matrix $nthreads U S N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

for matrix in ${MATRIX_FOLDER}/Original/Unsymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		./runtimeAnalysis $matrix $nthreads U N N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

echo "******************* CASE-2: METIS ******************"
for matrix in ${MATRIX_FOLDER}/Original/Symmetric/*.mtx; do
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		./runtimeAnalysis $matrix $nthreads U Y Y
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

for matrix in ${MATRIX_FOLDER}/Original/PatternSymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		./runtimeAnalysis $matrix $nthreads U S Y
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

for matrix in ${MATRIX_FOLDER}/Original/Unsymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		./runtimeAnalysis $matrix $nthreads U N Y
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done
echo "******************* BENCHMARK END ******************"
