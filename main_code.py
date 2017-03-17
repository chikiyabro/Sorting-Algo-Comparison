from random import randint
import numpy as np
import matplotlib.pyplot as plt
import datetime

comparisons1 = 0
comparisons2 = 0
comparisons3 = 0
swaps1 = 0
swaps2 = 0
swaps3 = 0

## Algorithms implemented using the psuedocodes from the book ##

###################################          to merge small arrays      ########################################################
def merge(left_half, right_half,comparisons,swaps):
    comparisons +=1
    if len(left_half) == 0 or len(right_half) == 0:
        return (left_half,comparisons,swaps) or (right_half,comparisons,swaps)

    merged = list()
    i = 0
    j = 0
    while len(merged) < (len(left_half) + len(right_half)):
        comparisons+=1
        if left_half[i] < right_half[j]:
            merged.append(left_half[i])
            i += 1
        else:
            merged.append(right_half[j])
            j += 1
        comparisons+=2
        swaps+=1
        if i == len(left_half) or j == len(right_half):
            merged.extend(left_half[i:] or right_half[j:])
            swaps += len(left_half[i:]) or len(right_half[j:])
            break

    return (merged,comparisons,swaps)


################################################################################################################################

###################################         Merge Sort Iterative Algorithm          ############################################
def iterativeMergeSort(arr):
    comparisons = 0
    swaps = 0
    subparts = [[x] for x in arr]
    while len(subparts) > 1:
        comparisons+=1
        temp = list()
        for i in range(0, len(subparts) // 2):
            comparisons+=1
            (r, comparisons, swaps) = merge(subparts[i * 2], subparts[i * 2 + 1], comparisons, swaps)
            temp.append(r)
            swaps+=1

        comparisons+=1
        if len(subparts) % 2:
            temp.append(subparts[-1])
            swaps+=1
        subparts = temp
    return (subparts[0],comparisons,swaps)


################################################################################################################################

###################################     Merge Sort Recursive Algorithm      ####################################################
def recursiveMergeSort(arr):
    global comparisons1, swaps1

    comparisons1+=1
    if len(arr) <= 1:
        return (arr,comparisons1,swaps1)

    middle = int(len(arr) / 2)
    (left_half,comparisons1,swaps1) = recursiveMergeSort(arr[:middle])
    (right_half,comparisons1,swaps1) = recursiveMergeSort(arr[middle:])
    (r,comparisons1,swaps1) = merge(left_half, right_half,comparisons1,swaps1)
    return (r,comparisons1,swaps1)


################################################################################################################################

###################################         Max Heapify Function           #####################################################
def maxHeapify(A, i, size,comparisons,swaps):
    left_half = 2 * i + 1
    right_half = 2 * i + 2
    largest = i
    if left_half < size and A[left_half] > A[largest]:
        comparisons+=2
        largest = left_half

    if right_half < size and A[right_half] > A[largest]:
        comparisons += 2
        largest = right_half
    if largest != i:
        comparisons += 1
        A[i], A[largest] = A[largest], A[i]
        swaps+=1
        (comparisons,swaps) = maxHeapify(A, largest, size,comparisons,swaps)
    return (comparisons,swaps)

################################################################################################################################

###################################         Building Max Heap Function      ####################################################
def buildMaxHeap(array,comparisons,swaps):
    length = len(array)
    for i in range(length // 2, -1, -1):
        comparisons+=1
        (comparisons,swaps) = maxHeapify(array, i, len(array),comparisons,swaps)
    return (comparisons,swaps)


################################################################################################################################

###################################         Heap Sort Algorithm         ########################################################
def heapSort(arr):
    comparisons = 0
    swaps = 0
    (comparisons,swaps) = buildMaxHeap(arr,comparisons,swaps)
    size = len(arr)
    for i in range(size - 1, 0, -1):
        comparisons+=1
        arr[0], arr[i] = arr[i], arr[0]
        swaps+=1
        size -= 1
        (comparisons,swaps) = maxHeapify(arr, 0, size,comparisons,swaps)
    return (comparisons,swaps)


################################################################################################################################

###################################         Insertion Sort Algorithm        ####################################################
def insertionSort(arr):
    comparisons = 0
    swaps = 0
    for i in range(1, len(arr)):
        comparisons+=1
        key = arr[i]
        j = i
        while j > 0 and key < arr[j - 1]:
            comparisons+=2
            arr[j] = arr[j - 1]
            swaps+=1
            j -= 1
        arr[j] = key
        swaps+=1
    return (comparisons,swaps)

################################################################################################################################

####################################     Partition function used in Deterministic Quick Sort ###################################
def deterministicPartition(arr, start, end ,comparisons , swaps):
    pivot = arr[end]
    i = start - 1
    for j in range(start, end):
        comparisons+=1
        if arr[j] <= pivot:
            comparisons+=1
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            swaps+=1
    arr[i + 1], arr[end] = arr[end], arr[i + 1]
    swaps+=1
    return (i+1 ,comparisons,swaps)


################################################################################################################################

###################################      Partition function used in Randomized Quick Sort   ####################################
def randomizedPartition(arr, start, end,comparisons , swaps):

    random_number = randint(start, end)
    arr[random_number], arr[end] = arr[end], arr[random_number]
    swaps+=1
    return deterministicPartition(arr, start, end,comparisons,swaps)


################################################################################################################################

###################################      Quick Sort Algorithm [Last Index as Pivot]        #####################################
def deterministicQuickSort(arr, start, end):
    global comparisons2,swaps2

    if start < end:
        comparisons2+=1
        (mid,comparisons2,swaps2) = deterministicPartition(arr, start, end,comparisons2,swaps2)
        (comparisons2,swaps2) = deterministicQuickSort(arr, start, mid - 1)
        (comparisons2, swaps2) = deterministicQuickSort(arr, mid + 1, end)
    return (comparisons2,swaps2)


################################################################################################################################

###################################          Quick Sort Algorithm [Random Pivot]        ########################################
def randomizedQuickSort(arr, start, end):
    global comparisons3 , swaps3

    if start < end:
        comparisons3+=1
        (mid, comparisons3, swaps3) = randomizedPartition(arr, start, end,comparisons3,swaps3)
        (comparisons3, swaps3) = randomizedQuickSort(arr, start, mid - 1)
        (comparisons3, swaps3) = randomizedQuickSort(arr, mid + 1, end)
    return (comparisons3, swaps3)

################################################################################################################################




def initialize_lists(arr_size):
    test = list()
    for i in range(0, arr_size):
        test.append(randint(0, 2 ** 32 - 1))
    return (test)

def sorting_algorithms(algo,arr_size):
    array_size = arr_size

    final_result=0
    final_swaps=0
    final_comparisons=0

    # print("Time Taken By:")
    if algo=="Merge Sort [Recursive]":

        start = datetime.datetime.now()
        (list1, comparisons, swaps) = recursiveMergeSort(initialize_lists(array_size))
        end = datetime.datetime.now()
        # print("\tMerge Sort [Recursive]:\t\t" + str(end - start) + "   " + str(comparisons) + "   " + str(swaps))
        final_result= (end - start).total_seconds()
        final_comparisons = comparisons
        final_swaps = swaps

    elif algo == "Merge Sort [Iterative]":
        start = datetime.datetime.now()
        (list1, comparisons, swaps) = iterativeMergeSort(initialize_lists(array_size))
        end = datetime.datetime.now()
        # print("\tMerge Sort [Iterative]:\t\t" + str(end - start) + "   " + str(comparisons) + "   " + str(swaps))
        final_result = (end - start).total_seconds()
        final_comparisons = comparisons
        final_swaps = swaps

    elif algo == "Quick Sort [Deterministic]":
        start = datetime.datetime.now()
        test_2 = initialize_lists(array_size)
        (comparisons, swaps) = deterministicQuickSort(test_2, 0, len(test_2) - 1)
        end = datetime.datetime.now()
        # print("\tQuick Sort [Deterministic]:\t" + str(end - start) + "   " + str(comparisons) + "   " + str(swaps))
        final_result = (end - start).total_seconds()
        final_comparisons = comparisons
        final_swaps = swaps

    elif algo == "Quick Sort [Randomized]":
        start = datetime.datetime.now()
        test_3 = initialize_lists(array_size)
        (comparisons, swaps) = randomizedQuickSort(test_3, 0, len(test_3) - 1)
        end = datetime.datetime.now()
        # print("\tQuick Sort [Randomized]:\t" + str(end - start) + "   " + str(comparisons) + "   " + str(swaps))
        final_result = (end - start).total_seconds()
        final_comparisons = comparisons
        final_swaps = swaps

    elif algo == "Heap Sort":
        start = datetime.datetime.now()
        (comparisons, swaps) = heapSort(initialize_lists(array_size))
        end = datetime.datetime.now()
        # print("\tHeap Sort:\t\t\t\t\t" + str(end - start) + "   " + str(comparisons) + "   " + str(swaps))
        final_result = (end - start).total_seconds()
        final_comparisons= comparisons
        final_swaps= swaps

    elif algo == "Insertion Sort":
        start = datetime.datetime.now()
        (comparisons, swaps) = insertionSort(initialize_lists(array_size))
        end = datetime.datetime.now()
        # print("\tInsertion Sort:\t\t\t\t" + str(end - start) + "   " + str(comparisons) + "   " + str(swaps))
        final_result = (end - start).total_seconds()
        final_comparisons = comparisons
        final_swaps = swaps

    return (final_result,final_comparisons,final_swaps)



heap_sort_time = {}
heap_sort_comparisons = {}
heap_sort_swaps = {}
sizes = [100,1000,10000,100000,1000000]
for k in range(5):
    (a,b,c) = sorting_algorithms("Heap Sort",sizes[k])

    heap_sort_time[str(sizes[k])] = a
    heap_sort_comparisons[str(sizes[k])] = b
    heap_sort_swaps[str(sizes[k])] = c

print(heap_sort_comparisons)

merge_sort_recursuve_time = {}
merge_sort_recursuve_comparisons = {}
merge_sort_recursuve_swaps = {}

for k in range(5):
    (a,b,c) = sorting_algorithms("Merge Sort [Recursive]",sizes[k])

    merge_sort_recursuve_time[str(sizes[k])] = a
    merge_sort_recursuve_comparisons[str(sizes[k])] = b
    merge_sort_recursuve_swaps[str(sizes[k])] = c

print(merge_sort_recursuve_comparisons)
#
merge_sort_Iterative_time = {}
merge_sort_Iterative_comparisons = {}
merge_sort_Iterative_swaps = {}

for k in range(5):
    (a,b,c) = sorting_algorithms("Merge Sort [Iterative]",sizes[k])

    merge_sort_Iterative_time[str(sizes[k])] = a
    merge_sort_Iterative_comparisons[str(sizes[k])] = b
    merge_sort_Iterative_swaps[str(sizes[k])] = c

print(merge_sort_Iterative_comparisons)

quick_sort_deterministic_time = {}
quick_sort_deterministic_comparisons = {}
quick_sort_deterministic_swaps = {}

for k in range(5):
    (a,b,c) = sorting_algorithms("Quick Sort [Deterministic]",sizes[k])

    quick_sort_deterministic_time[str(sizes[k])] = a
    quick_sort_deterministic_comparisons[str(sizes[k])] = b
    quick_sort_deterministic_swaps[str(sizes[k])] = c

print(quick_sort_deterministic_comparisons)


quick_sort_Randomized_time = {}
quick_sort_Randomized_comparisons = {}
quick_sort_Randomized_swaps = {}

for k in range(5):
    (a,b,c) = sorting_algorithms("Quick Sort [Randomized]",sizes[k])

    quick_sort_Randomized_time[str(sizes[k])] = a
    quick_sort_Randomized_comparisons[str(sizes[k])] = b
    quick_sort_Randomized_swaps[str(sizes[k])] = c

print(quick_sort_Randomized_comparisons)



insertion_sort_time = {}
insertion_sort_comparisons = {}
insertion_sort_swaps = {}
sizess = [100,1000,10000]
for k in range(3):
    (a,b,c) = sorting_algorithms("Insertion Sort",sizess[k] )

    insertion_sort_time[str(sizess[k])] = a
    insertion_sort_comparisons[str(sizess[k])] = b
    insertion_sort_swaps[str(sizess[k])] = c

print(insertion_sort_comparisons)

comparisons_graph = []
comparisons_graph.append(heap_sort_comparisons["100"])
comparisons_graph.append(heap_sort_comparisons["1000"])
comparisons_graph.append(heap_sort_comparisons["10000"])
comparisons_graph.append(heap_sort_comparisons["100000"])
comparisons_graph.append(heap_sort_comparisons["1000000"])

time_graph = []
time_graph.append(heap_sort_time["100"])
time_graph.append(heap_sort_time["1000"])
time_graph.append(heap_sort_time["10000"])
time_graph.append(heap_sort_time["100000"])
time_graph.append(heap_sort_time["1000000"])

Swaps_graph = []
Swaps_graph.append(heap_sort_swaps["100"])
Swaps_graph.append(heap_sort_swaps["1000"])
Swaps_graph.append(heap_sort_swaps["10000"])
Swaps_graph.append(heap_sort_swaps["100000"])
Swaps_graph.append(heap_sort_swaps["1000000"])

comparisons_graph1 = []
comparisons_graph1.append(merge_sort_recursuve_comparisons["100"])
comparisons_graph1.append(merge_sort_recursuve_comparisons["1000"])
comparisons_graph1.append(merge_sort_recursuve_comparisons["10000"])
comparisons_graph1.append(merge_sort_recursuve_comparisons["100000"])
comparisons_graph1.append(merge_sort_recursuve_comparisons["1000000"])

time_graph1 = []
time_graph1.append(merge_sort_recursuve_time["100"])
time_graph1.append(merge_sort_recursuve_time["1000"])
time_graph1.append(merge_sort_recursuve_time["10000"])
time_graph1.append(merge_sort_recursuve_time["100000"])
time_graph1.append(merge_sort_recursuve_time["1000000"])

Swaps_graph1 = []
Swaps_graph1.append(merge_sort_recursuve_swaps["100"])
Swaps_graph1.append(merge_sort_recursuve_swaps["1000"])
Swaps_graph1.append(merge_sort_recursuve_swaps["10000"])
Swaps_graph1.append(merge_sort_recursuve_swaps["100000"])
Swaps_graph1.append(merge_sort_recursuve_swaps["1000000"])


comparisons_graph2 = []
comparisons_graph2.append(merge_sort_Iterative_comparisons["100"])
comparisons_graph2.append(merge_sort_Iterative_comparisons["1000"])
comparisons_graph2.append(merge_sort_Iterative_comparisons["10000"])
comparisons_graph2.append(merge_sort_Iterative_comparisons["100000"])
comparisons_graph2.append(merge_sort_Iterative_comparisons["1000000"])

time_graph2 = []
time_graph2.append(merge_sort_Iterative_time["100"])
time_graph2.append(merge_sort_Iterative_time["1000"])
time_graph2.append(merge_sort_Iterative_time["10000"])
time_graph2.append(merge_sort_Iterative_time["100000"])
time_graph2.append(merge_sort_Iterative_time["1000000"])

Swaps_graph2 = []
Swaps_graph2.append(merge_sort_Iterative_swaps["100"])
Swaps_graph2.append(merge_sort_Iterative_swaps["1000"])
Swaps_graph2.append(merge_sort_Iterative_swaps["10000"])
Swaps_graph2.append(merge_sort_Iterative_swaps["100000"])
Swaps_graph2.append(merge_sort_Iterative_swaps["1000000"])

comparisons_graph3 = []
comparisons_graph3.append(quick_sort_deterministic_comparisons["100"])
comparisons_graph3.append(quick_sort_deterministic_comparisons["1000"])
comparisons_graph3.append(quick_sort_deterministic_comparisons["10000"])
comparisons_graph3.append(quick_sort_deterministic_comparisons["100000"])
comparisons_graph3.append(quick_sort_deterministic_comparisons["1000000"])

time_graph3 = []
time_graph3.append(quick_sort_deterministic_time["100"])
time_graph3.append(quick_sort_deterministic_time["1000"])
time_graph3.append(quick_sort_deterministic_time["10000"])
time_graph3.append(quick_sort_deterministic_time["100000"])
time_graph3.append(quick_sort_deterministic_time["1000000"])

Swaps_graph3 = []
Swaps_graph3.append(quick_sort_deterministic_swaps["100"])
Swaps_graph3.append(quick_sort_deterministic_swaps["1000"])
Swaps_graph3.append(quick_sort_deterministic_swaps["10000"])
Swaps_graph3.append(quick_sort_deterministic_swaps["100000"])
Swaps_graph3.append(quick_sort_deterministic_swaps["1000000"])


comparisons_graph4 = []
comparisons_graph4.append(quick_sort_Randomized_comparisons["100"])
comparisons_graph4.append(quick_sort_Randomized_comparisons["1000"])
comparisons_graph4.append(quick_sort_Randomized_comparisons["10000"])
comparisons_graph4.append(quick_sort_Randomized_comparisons["100000"])
comparisons_graph4.append(quick_sort_Randomized_comparisons["1000000"])

time_graph4 = []
time_graph4.append(quick_sort_Randomized_time["100"])
time_graph4.append(quick_sort_Randomized_time["1000"])
time_graph4.append(quick_sort_Randomized_time["10000"])
time_graph4.append(quick_sort_Randomized_time["100000"])
time_graph4.append(quick_sort_Randomized_time["1000000"])

Swaps_graph4 = []
Swaps_graph4.append(quick_sort_Randomized_swaps["100"])
Swaps_graph4.append(quick_sort_Randomized_swaps["1000"])
Swaps_graph4.append(quick_sort_Randomized_swaps["10000"])
Swaps_graph4.append(quick_sort_Randomized_swaps["100000"])
Swaps_graph4.append(quick_sort_Randomized_swaps["1000000"])


comparisons_graph5 = []
comparisons_graph5.append(insertion_sort_comparisons["100"])
comparisons_graph5.append(insertion_sort_comparisons["1000"])
comparisons_graph5.append(insertion_sort_comparisons["10000"])
comparisons_graph5.append(0)
comparisons_graph5.append(0)


time_graph5 = []
time_graph5.append(insertion_sort_time["100"])
time_graph5.append(insertion_sort_time["1000"])
time_graph5.append(insertion_sort_time["10000"])
time_graph5.append(0)
time_graph5.append(0)



Swaps_graph5 = []
Swaps_graph5.append(insertion_sort_swaps["100"])
Swaps_graph5.append(insertion_sort_swaps["1000"])
Swaps_graph5.append(insertion_sort_swaps["10000"])
Swaps_graph5.append(0)
Swaps_graph5.append(0)



















n_groups = 5
fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.1

opacity = 0.4
error_config = {'ecolor': '0.3'}


rects1 = plt.bar(index, comparisons_graph, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='Heap Sort')

rects2 = plt.bar(index + bar_width, comparisons_graph1, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='Merge Sort (Recursive)')

rects3 = plt.bar(index + bar_width+bar_width, comparisons_graph2, bar_width,
                 alpha=opacity,
                 color='g',
                 error_kw=error_config,
                 label='Merge Sort (Iterative)')

rects4 = plt.bar(index + 3*bar_width, comparisons_graph3, bar_width,
                 alpha=opacity,
                 color='c',
                 error_kw=error_config,
                 label='Quick Sort (Deterministic)')

rects5 = plt.bar(index + 4*bar_width, comparisons_graph4, bar_width,
                 alpha=opacity,
                 color='y',
                 error_kw=error_config,
                 label='Quick Sort (Randomized)')

rects6 = plt.bar(index +5* bar_width, comparisons_graph5, bar_width,
                 alpha=opacity,
                 color='m',
                 error_kw=error_config,
                 label='Insertion Sort ')

plt.xlabel('Input Array Sizes')
plt.ylabel('Values')
plt.title('No. of Comparisons for different sorting algorithms')
plt.xticks(index + bar_width / 2, ('100', '1000', '10000', '100000', '1000000'))
plt.legend()

plt.tight_layout()

plt.savefig('comparisons.png')



n_groups = 5
fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.1

opacity = 0.4
error_config = {'ecolor': '0.3'}


rects1 = plt.bar(index, time_graph, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='Heap Sort')

rects2 = plt.bar(index + bar_width, time_graph1, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='Merge Sort (Recursive)')

rects3 = plt.bar(index + bar_width+bar_width, time_graph2, bar_width,
                 alpha=opacity,
                 color='g',
                 error_kw=error_config,
                 label='Merge Sort (Iterative)')

rects4 = plt.bar(index + 3*bar_width, time_graph3, bar_width,
                 alpha=opacity,
                 color='c',
                 error_kw=error_config,
                 label='Quick Sort (Deterministic)')

rects5 = plt.bar(index + 4*bar_width, time_graph4, bar_width,
                 alpha=opacity,
                 color='y',
                 error_kw=error_config,
                 label='Quick Sort (Randomized)')

rects6 = plt.bar(index +5* bar_width, time_graph5, bar_width,
                 alpha=opacity,
                 color='m',
                 error_kw=error_config,
                 label='Insertion Sort ')

plt.xlabel('Input Array Sizes')
plt.ylabel('Values')
plt.title('Time Taken for different sorting algorithms')
plt.xticks(index + bar_width / 2, ('100', '1000', '10000', '100000', '1000000'))
plt.legend()

plt.tight_layout()

plt.savefig('time.png')



n_groups = 5
fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.1

opacity = 0.4
error_config = {'ecolor': '0.3'}


rects1 = plt.bar(index, Swaps_graph, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='Heap Sort')

rects2 = plt.bar(index + bar_width, Swaps_graph1, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='Merge Sort (Recursive)')

rects3 = plt.bar(index + bar_width+bar_width, Swaps_graph2, bar_width,
                 alpha=opacity,
                 color='g',
                 error_kw=error_config,
                 label='Merge Sort (Iterative)')

rects4 = plt.bar(index + 3*bar_width, Swaps_graph3, bar_width,
                 alpha=opacity,
                 color='c',
                 error_kw=error_config,
                 label='Quick Sort (Deterministic)')

rects5 = plt.bar(index + 4*bar_width, Swaps_graph4, bar_width,
                 alpha=opacity,
                 color='y',
                 error_kw=error_config,
                 label='Quick Sort (Randomized)')

rects6 = plt.bar(index +5* bar_width, Swaps_graph5, bar_width,
                 alpha=opacity,
                 color='m',
                 error_kw=error_config,
                 label='Insertion Sort ')

plt.xlabel('Input Array Sizes')
plt.ylabel('Values')
plt.title('No. of Swaps for different sorting algorithms')
plt.xticks(index + bar_width / 2, ('100', '1000', '10000', '100000', '1000000'))
plt.legend()

plt.tight_layout()

plt.savefig('swaps.png')