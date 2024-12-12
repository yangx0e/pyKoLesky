from pyKoLesky.maxheap import *

# Driver Code
if __name__ == "__main__":
    print('The maxHeap is ')

    dists = [5.0, 3.0, 17.0, 10.0, 84.0, 19.0, 6.0, 22.0, 9.0, 20, 84]
    maxHeap = MaxHeap(dists)
    # Print the heap structure
    maxHeap.Print()
    # Extract and print the max element
    max_element = maxHeap.pop()
    print(f"\nThe Max val is (index: {max_element.index}, l: {max_element.l})")
    print("\n")
    maxHeap.Print()