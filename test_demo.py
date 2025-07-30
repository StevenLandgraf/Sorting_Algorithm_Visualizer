"""
Demo script to test the sorting algorithms without Streamlit
"""

from app import SortingVisualizer

def test_algorithms():
    """Test all sorting algorithms with a small array"""
    visualizer = SortingVisualizer()
    test_array = [64, 34, 25, 12, 22, 11, 90]
    
    algorithms = {
        "Bubble Sort": visualizer.bubble_sort,
        "Selection Sort": visualizer.selection_sort,
        "Insertion Sort": visualizer.insertion_sort,
        "Merge Sort": visualizer.merge_sort,
        "Quick Sort": visualizer.quick_sort
    }
    
    print("🎯 Testing Sorting Algorithms")
    print("=" * 50)
    print(f"Original array: {test_array}")
    print()
    
    for name, algorithm in algorithms.items():
        print(f"Testing {name}...")
        steps = list(algorithm(test_array))
        final_array = steps[-1][0]
        print(f"✅ {name}: {final_array} (Steps: {len(steps)})")
        
        # Verify sorting
        is_sorted = all(final_array[i] <= final_array[i+1] for i in range(len(final_array)-1))
        print(f"   Correctly sorted: {'✅ Yes' if is_sorted else '❌ No'}")
        print()

if __name__ == "__main__":
    test_algorithms()
