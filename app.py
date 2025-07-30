import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
from typing import List, Tuple, Generator

# Set page configuration
st.set_page_config(
    page_title="Sorting Algorithm Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SortingVisualizer:
    def __init__(self):
        self.array = []
        self.steps = []
    
    def generate_array(self, size: int, min_val: int = 1, max_val: int = 100) -> List[int]:
        """Generate a random array of given size"""
        return [random.randint(min_val, max_val) for _ in range(size)]
    
    def bubble_sort(self, arr: List[int]) -> Generator[Tuple[List[int], int, int, str], None, None]:
        """Bubble sort with step-by-step visualization data"""
        arr = arr.copy()
        n = len(arr)
        
        for i in range(n):
            for j in range(0, n - i - 1):
                # Yield current state with indices being compared
                yield arr.copy(), j, j + 1, f"Comparing elements at positions {j} and {j+1}"
                
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    yield arr.copy(), j, j + 1, f"Swapped elements at positions {j} and {j+1}"
        
        yield arr.copy(), -1, -1, "Sorting complete!"
    
    def selection_sort(self, arr: List[int]) -> Generator[Tuple[List[int], int, int, str], None, None]:
        """Selection sort with step-by-step visualization data"""
        arr = arr.copy()
        n = len(arr)
        
        for i in range(n):
            min_idx = i
            yield arr.copy(), i, min_idx, f"Finding minimum element from position {i}"
            
            for j in range(i + 1, n):
                yield arr.copy(), j, min_idx, f"Comparing element at position {j} with current minimum"
                if arr[j] < arr[min_idx]:
                    min_idx = j
                    yield arr.copy(), j, min_idx, f"New minimum found at position {j}"
            
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
                yield arr.copy(), i, min_idx, f"Swapped elements at positions {i} and {min_idx}"
        
        yield arr.copy(), -1, -1, "Sorting complete!"
    
    def insertion_sort(self, arr: List[int]) -> Generator[Tuple[List[int], int, int, str], None, None]:
        """Insertion sort with step-by-step visualization data"""
        arr = arr.copy()
        
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            
            yield arr.copy(), i, j, f"Inserting element {key} at position {i}"
            
            while j >= 0 and arr[j] > key:
                yield arr.copy(), j, j + 1, f"Moving element {arr[j]} to the right"
                arr[j + 1] = arr[j]
                j -= 1
                yield arr.copy(), j + 1, j + 2, f"Element moved"
            
            arr[j + 1] = key
            yield arr.copy(), j + 1, i, f"Inserted element {key} at position {j + 1}"
        
        yield arr.copy(), -1, -1, "Sorting complete!"
    
    def merge_sort(self, arr: List[int]) -> Generator[Tuple[List[int], int, int, str], None, None]:
        """Merge sort with step-by-step visualization data"""
        def merge_sort_helper(arr, left, right, original_arr):
            if left < right:
                mid = (left + right) // 2
                
                yield from merge_sort_helper(arr, left, mid, original_arr)
                yield from merge_sort_helper(arr, mid + 1, right, original_arr)
                yield from merge(arr, left, mid, right, original_arr)
        
        def merge(arr, left, mid, right, original_arr):
            left_arr = arr[left:mid + 1]
            right_arr = arr[mid + 1:right + 1]
            
            i = j = 0
            k = left
            
            while i < len(left_arr) and j < len(right_arr):
                yield original_arr.copy(), left + i, mid + 1 + j, f"Merging subarrays: comparing {left_arr[i]} and {right_arr[j]}"
                
                if left_arr[i] <= right_arr[j]:
                    original_arr[k] = left_arr[i]
                    i += 1
                else:
                    original_arr[k] = right_arr[j]
                    j += 1
                k += 1
                
                yield original_arr.copy(), k - 1, -1, f"Placed element {original_arr[k-1]} at position {k-1}"
            
            while i < len(left_arr):
                original_arr[k] = left_arr[i]
                yield original_arr.copy(), k, -1, f"Copying remaining element {left_arr[i]}"
                i += 1
                k += 1
            
            while j < len(right_arr):
                original_arr[k] = right_arr[j]
                yield original_arr.copy(), k, -1, f"Copying remaining element {right_arr[j]}"
                j += 1
                k += 1
        
        arr_copy = arr.copy()
        yield from merge_sort_helper(arr_copy, 0, len(arr_copy) - 1, arr_copy)
        yield arr_copy, -1, -1, "Sorting complete!"
    
    def quick_sort(self, arr: List[int]) -> Generator[Tuple[List[int], int, int, str], None, None]:
        """Quick sort with step-by-step visualization data"""
        def quick_sort_helper(arr, low, high):
            if low < high:
                pi = yield from partition(arr, low, high)
                yield from quick_sort_helper(arr, low, pi - 1)
                yield from quick_sort_helper(arr, pi + 1, high)
        
        def partition(arr, low, high):
            pivot = arr[high]
            yield arr.copy(), high, -1, f"Pivot selected: {pivot} at position {high}"
            
            i = low - 1
            
            for j in range(low, high):
                yield arr.copy(), j, high, f"Comparing {arr[j]} with pivot {pivot}"
                
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
                    yield arr.copy(), i, j, f"Swapped {arr[i]} and {arr[j]}"
            
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            yield arr.copy(), i + 1, high, f"Placed pivot {pivot} at position {i + 1}"
            
            return i + 1
        
        arr_copy = arr.copy()
        yield from quick_sort_helper(arr_copy, 0, len(arr_copy) - 1)
        yield arr_copy, -1, -1, "Sorting complete!"

def create_bar_chart(array: List[int], highlight_indices: Tuple[int, int] = (-1, -1), 
                    title: str = "Array Visualization") -> go.Figure:
    """Create a bar chart visualization of the array"""
    colors = ['lightblue'] * len(array)
    
    # Highlight specific indices
    if highlight_indices[0] >= 0 and highlight_indices[0] < len(colors):
        colors[highlight_indices[0]] = 'red'
    if highlight_indices[1] >= 0 and highlight_indices[1] < len(colors):
        colors[highlight_indices[1]] = 'orange'
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(range(len(array))),
            y=array,
            marker_color=colors,
            text=array,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Index",
        yaxis_title="Value",
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    st.title("🎯 Sorting Algorithm Visualizer")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    
    # Array configuration
    array_size = st.sidebar.slider("Array Size", min_value=5, max_value=50, value=20)
    min_value = st.sidebar.number_input("Minimum Value", value=1, min_value=1)
    max_value = st.sidebar.number_input("Maximum Value", value=100, min_value=int(min_value))
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Select Sorting Algorithm",
        ["Bubble Sort", "Selection Sort", "Insertion Sort", "Merge Sort", "Quick Sort"]
    )
    
    # Animation speed
    animation_speed = st.sidebar.slider("Animation Speed (seconds)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    
    # Initialize visualizer
    visualizer = SortingVisualizer()
    
    # Generate new array button
    if st.sidebar.button("🔄 Generate New Array"):
        st.session_state.array = visualizer.generate_array(array_size, min_value, max_value)
        st.session_state.sorted = False
        st.session_state.steps = []
    
    # Initialize array if not exists
    if 'array' not in st.session_state:
        st.session_state.array = visualizer.generate_array(array_size, min_value, max_value)
        st.session_state.sorted = False
        st.session_state.steps = []
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 Current Array - {algorithm}")
        
        # Display current array
        chart_placeholder = st.empty()
        
        # Show initial array
        if not st.session_state.get('sorting', False):
            fig = create_bar_chart(st.session_state.array, title=f"Initial Array ({algorithm})")
            chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Algorithm Info")
        
        # Algorithm information
        algorithm_info = {
            "Bubble Sort": {
                "Time Complexity": "O(n²)",
                "Space Complexity": "O(1)",
                "Description": "Compares adjacent elements and swaps them if they're in wrong order."
            },
            "Selection Sort": {
                "Time Complexity": "O(n²)",
                "Space Complexity": "O(1)",
                "Description": "Finds minimum element and places it at the beginning."
            },
            "Insertion Sort": {
                "Time Complexity": "O(n²)",
                "Space Complexity": "O(1)",
                "Description": "Builds sorted array one element at a time."
            },
            "Merge Sort": {
                "Time Complexity": "O(n log n)",
                "Space Complexity": "O(n)",
                "Description": "Divides array and merges sorted halves."
            },
            "Quick Sort": {
                "Time Complexity": "O(n log n) avg",
                "Space Complexity": "O(log n)",
                "Description": "Partitions array around pivot element."
            }
        }
        
        info = algorithm_info[algorithm]
        st.metric("Time Complexity", info["Time Complexity"])
        st.metric("Space Complexity", info["Space Complexity"])
        st.write("**Description:**")
        st.write(info["Description"])
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_sorting = st.button("▶️ Start Sorting", disabled=st.session_state.get('sorting', False))
    
    with col2:
        reset_array = st.button("🔄 Reset Array")
    
    with col3:
        st.write(f"**Array Status:** {'✅ Sorted' if st.session_state.get('sorted', False) else '❌ Unsorted'}")
    
    # Reset array
    if reset_array:
        st.session_state.array = visualizer.generate_array(array_size, min_value, max_value)
        st.session_state.sorted = False
        st.session_state.sorting = False
        st.session_state.steps = []
        st.rerun()
    
    # Start sorting animation
    if start_sorting:
        st.session_state.sorting = True
        
        # Status placeholder
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Get the appropriate sorting algorithm
        if algorithm == "Bubble Sort":
            sort_generator = visualizer.bubble_sort(st.session_state.array)
        elif algorithm == "Selection Sort":
            sort_generator = visualizer.selection_sort(st.session_state.array)
        elif algorithm == "Insertion Sort":
            sort_generator = visualizer.insertion_sort(st.session_state.array)
        elif algorithm == "Merge Sort":
            sort_generator = visualizer.merge_sort(st.session_state.array)
        elif algorithm == "Quick Sort":
            sort_generator = visualizer.quick_sort(st.session_state.array)
        
        # Animate sorting steps
        steps = list(sort_generator)
        total_steps = len(steps)
        
        for i, (current_array, idx1, idx2, description) in enumerate(steps):
            # Update progress
            progress = (i + 1) / total_steps
            progress_bar.progress(progress)
            
            # Update status
            status_placeholder.write(f"**Step {i + 1}/{total_steps}:** {description}")
            
            # Update visualization
            fig = create_bar_chart(
                current_array, 
                highlight_indices=(idx1, idx2),
                title=f"{algorithm} - Step {i + 1}"
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Add delay for animation
            time.sleep(animation_speed)
        
        # Update session state
        st.session_state.array = steps[-1][0]  # Final sorted array
        st.session_state.sorted = True
        st.session_state.sorting = False
        
        status_placeholder.success("🎉 Sorting completed successfully!")
        progress_bar.progress(1.0)
    
    # Show array statistics
    st.markdown("---")
    st.subheader("📊 Array Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Array Size", len(st.session_state.array))
    
    with col2:
        st.metric("Minimum Value", min(st.session_state.array))
    
    with col3:
        st.metric("Maximum Value", max(st.session_state.array))
    
    with col4:
        st.metric("Sum", sum(st.session_state.array))
    
    # Show raw array data
    with st.expander("🔍 View Raw Array Data"):
        st.write("**Current Array:**")
        st.write(st.session_state.array)

if __name__ == "__main__":
    main()
