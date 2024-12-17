
#!/bin/sh
MATRIX_FOLDER=/home/paul/UNSA/directs/spike_pstrsv/exs
RESULTS_FOLDER=/home/paul/UNSA/directs/spike_pstrsv/results
mkdir -p $RESULTS_FOLDER

echo "******************* CASE-1: ORIGINAL ******************" | tee -a $RESULTS_FOLDER/benchmark_results.txt

for matrix in ${MATRIX_FOLDER}/Original/Symmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######" | tee -a $RESULTS_FOLDER/benchmark_results.txt
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads | tee -a $RESULTS_FOLDER/benchmark_results.txt
		numactl --cpunodebind=0 ./runtimeAnalysis $matrix $nthreads u y n | tee -a $RESULTS_FOLDER/benchmark_results.txt
	done
	echo "######-END: Tests on "$matrix"-######" | tee -a $RESULTS_FOLDER/benchmark_results.txt
	echo "" | tee -a $RESULTS_FOLDER/benchmark_results.txt
done

for matrix in ${MATRIX_FOLDER}/Original/PatternSymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######" | tee -a $RESULTS_FOLDER/benchmark_results.txt
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads | tee -a $RESULTS_FOLDER/benchmark_results.txt
		numactl --cpunodebind=0 ./runtimeAnalysis $matrix $nthreads u s n | tee -a $RESULTS_FOLDER/benchmark_results.txt
	done
	echo "######-END: Tests on "$matrix"-######" | tee -a $RESULTS_FOLDER/benchmark_results.txt
	echo "" | tee -a $RESULTS_FOLDER/benchmark_results.txt
done

for matrix in ${MATRIX_FOLDER}/Original/Unsymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######" | tee -a $RESULTS_FOLDER/benchmark_results.txt
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads | tee -a $RESULTS_FOLDER/benchmark_results.txt
		numactl --cpunodebind=0 ./runtimeAnalysis $matrix $nthreads u n n | tee -a $RESULTS_FOLDER/benchmark_results.txt
	done
	echo "######-END: Tests on "$matrix"-######" | tee -a $RESULTS_FOLDER/benchmark_results.txt
	echo "" | tee -a $RESULTS_FOLDER/benchmark_results.txt
done

echo "******************* CASE-2: METIS ******************" | tee -a $RESULTS_FOLDER/benchmark_results.txt

for matrix in ${MATRIX_FOLDER}/Original/Symmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######" | tee -a $RESULTS_FOLDER/benchmark_results.txt
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads | tee -a $RESULTS_FOLDER/benchmark_results.txt
		numactl --cpunodebind=0 ./runtimeAnalysis $matrix $nthreads u y y | tee -a $RESULTS_FOLDER/benchmark_results.txt
	done
	echo "######-END: Tests on "$matrix"-######" | tee -a $RESULTS_FOLDER/benchmark_results.txt
	echo "" | tee -a $RESULTS_FOLDER/benchmark_results.txt
done

for matrix in ${MATRIX_FOLDER}/Original/PatternSymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######" | tee -a $RESULTS_FOLDER/benchmark_results.txt
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads | tee -a $RESULTS_FOLDER/benchmark_results.txt
		numactl --cpunodebind=0 ./runtimeAnalysis $matrix $nthreads u s y | tee -a $RESULTS_FOLDER/benchmark_results.txt
	done
	echo "######-END: Tests on "$matrix"-######" | tee -a $RESULTS_FOLDER/benchmark_results.txt
	echo "" | tee -a $RESULTS_FOLDER/benchmark_results.txt
done

for matrix in ${MATRIX_FOLDER}/Original/Unsymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######" | tee -a $RESULTS_FOLDER/benchmark_results.txt
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads | tee -a $RESULTS_FOLDER/benchmark_results.txt
		numactl --cpunodebind=0 ./runtimeAnalysis $matrix $nthreads u n y | tee -a $RESULTS_FOLDER/benchmark_results.txt
	done
	echo "######-END: Tests on "$matrix"-######" | tee -a $RESULTS_FOLDER/benchmark_results.txt
	echo "" | tee -a $RESULTS_FOLDER/benchmark_results.txt
done

echo "******************* BENCHMARK END ******************" | tee -a $RESULTS_FOLDER/benchmark_results.txt

