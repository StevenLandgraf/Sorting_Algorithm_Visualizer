import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
from typing import List, Tuple, Generator
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Advanced Sorting Visualizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvancedSortingVisualizer:
    def __init__(self):
        self.comparison_count = 0
        self.swap_count = 0
        self.array_accesses = 0
    
    def reset_counters(self):
        """Reset all performance counters"""
        self.comparison_count = 0
        self.swap_count = 0
        self.array_accesses = 0
    
    def generate_array(self, size: int, array_type: str = "random") -> List[int]:
        """Generate different types of arrays for testing"""
        if array_type == "random":
            return [random.randint(1, 100) for _ in range(size)]
        elif array_type == "nearly_sorted":
            arr = list(range(1, size + 1))
            # Swap a few elements to make it nearly sorted
            for _ in range(size // 10):
                i, j = random.randint(0, size-1), random.randint(0, size-1)
                arr[i], arr[j] = arr[j], arr[i]
            return arr
        elif array_type == "reverse_sorted":
            return list(range(size, 0, -1))
        elif array_type == "sorted":
            return list(range(1, size + 1))
        elif array_type == "duplicates":
            return [random.randint(1, size//4) for _ in range(size)]
    
    def heapsort(self, arr: List[int]) -> Generator[Tuple[List[int], int, int, str, dict], None, None]:
        """Heap sort with step-by-step visualization"""
        def heapify(arr, n, i):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            self.comparison_count += 1
            if left < n and arr[left] > arr[largest]:
                largest = left
            
            self.comparison_count += 1
            if right < n and arr[right] > arr[largest]:
                largest = right
            
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.swap_count += 1
                self.array_accesses += 2
                yield arr.copy(), i, largest, f"Heapifying: swapped {arr[i]} and {arr[largest]}", self.get_stats()
                yield from heapify(arr, n, largest)
        
        arr = arr.copy()
        n = len(arr)
        
        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            yield from heapify(arr, n, i)
        
        # Extract elements from heap one by one
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            self.swap_count += 1
            self.array_accesses += 2
            yield arr.copy(), 0, i, f"Moved max element {arr[i]} to position {i}", self.get_stats()
            yield from heapify(arr, i, 0)
        
        yield arr.copy(), -1, -1, "Heap sort complete!", self.get_stats()
    
    def cocktail_sort(self, arr: List[int]) -> Generator[Tuple[List[int], int, int, str, dict], None, None]:
        """Cocktail shaker sort with step-by-step visualization"""
        arr = arr.copy()
        n = len(arr)
        swapped = True
        start = 0
        end = n - 1
        
        while swapped:
            swapped = False
            
            # Forward pass
            for i in range(start, end):
                self.comparison_count += 1
                self.array_accesses += 2
                yield arr.copy(), i, i + 1, f"Forward pass: comparing {arr[i]} and {arr[i+1]}", self.get_stats()
                
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    self.swap_count += 1
                    swapped = True
                    yield arr.copy(), i, i + 1, f"Forward pass: swapped {arr[i+1]} and {arr[i]}", self.get_stats()
            
            if not swapped:
                break
            
            end -= 1
            swapped = False
            
            # Backward pass
            for i in range(end - 1, start - 1, -1):
                self.comparison_count += 1
                self.array_accesses += 2
                yield arr.copy(), i, i + 1, f"Backward pass: comparing {arr[i]} and {arr[i+1]}", self.get_stats()
                
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    self.swap_count += 1
                    swapped = True
                    yield arr.copy(), i, i + 1, f"Backward pass: swapped {arr[i+1]} and {arr[i]}", self.get_stats()
            
            start += 1
        
        yield arr.copy(), -1, -1, "Cocktail sort complete!", self.get_stats()
    
    def shell_sort(self, arr: List[int]) -> Generator[Tuple[List[int], int, int, str, dict], None, None]:
        """Shell sort with step-by-step visualization"""
        arr = arr.copy()
        n = len(arr)
        gap = n // 2
        
        while gap > 0:
            yield arr.copy(), -1, -1, f"Using gap size: {gap}", self.get_stats()
            
            for i in range(gap, n):
                temp = arr[i]
                j = i
                
                while j >= gap and arr[j - gap] > temp:
                    self.comparison_count += 1
                    self.array_accesses += 2
                    yield arr.copy(), j, j - gap, f"Gap {gap}: comparing {arr[j-gap]} and {temp}", self.get_stats()
                    
                    arr[j] = arr[j - gap]
                    self.swap_count += 1
                    j -= gap
                    yield arr.copy(), j + gap, j, f"Gap {gap}: moved {arr[j+gap]} to position {j+gap}", self.get_stats()
                
                arr[j] = temp
                self.array_accesses += 1
                yield arr.copy(), j, i, f"Gap {gap}: inserted {temp} at position {j}", self.get_stats()
            
            gap //= 2
        
        yield arr.copy(), -1, -1, "Shell sort complete!", self.get_stats()
    
    def get_stats(self) -> dict:
        """Get current performance statistics"""
        return {
            "comparisons": self.comparison_count,
            "swaps": self.swap_count,
            "array_accesses": self.array_accesses
        }

def create_advanced_chart(array: List[int], highlight_indices: Tuple[int, int] = (-1, -1),
                         title: str = "Array Visualization", stats: dict = None) -> go.Figure:
    """Create an advanced bar chart with additional information"""
    colors = ['lightblue'] * len(array)
    
    # Highlight specific indices
    if highlight_indices[0] >= 0 and highlight_indices[0] < len(colors):
        colors[highlight_indices[0]] = 'red'
    if highlight_indices[1] >= 0 and highlight_indices[1] < len(colors):
        colors[highlight_indices[1]] = 'orange'
    
    # Create subplot with secondary y-axis for statistics
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.8, 0.2],
        subplot_titles=[title, "Performance Metrics"],
        vertical_spacing=0.1
    )
    
    # Main bar chart
    fig.add_trace(
        go.Bar(
            x=list(range(len(array))),
            y=array,
            marker_color=colors,
            text=array,
            textposition='outside',
            name="Array Elements"
        ),
        row=1, col=1
    )
    
    # Performance metrics
    if stats:
        metrics = ['Comparisons', 'Swaps', 'Array Accesses']
        values = [stats['comparisons'], stats['swaps'], stats['array_accesses']]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                marker_color=['lightcoral', 'lightgreen', 'lightyellow'],
                text=values,
                textposition='outside',
                name="Performance"
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        xaxis_title="Index",
        yaxis_title="Value"
    )
    
    return fig

def compare_algorithms(array: List[int], algorithms: List[str]) -> pd.DataFrame:
    """Compare performance of different algorithms"""
    results = []
    
    for algorithm in algorithms:
        visualizer = AdvancedSortingVisualizer()
        visualizer.reset_counters()
        
        start_time = time.time()
        
        if algorithm == "Heap Sort":
            steps = list(visualizer.heapsort(array))
        elif algorithm == "Cocktail Sort":
            steps = list(visualizer.cocktail_sort(array))
        elif algorithm == "Shell Sort":
            steps = list(visualizer.shell_sort(array))
        
        end_time = time.time()
        
        final_stats = steps[-1][4] if len(steps) > 0 else {"comparisons": 0, "swaps": 0, "array_accesses": 0}
        
        results.append({
            "Algorithm": algorithm,
            "Time (seconds)": round(end_time - start_time, 4),
            "Comparisons": final_stats["comparisons"],
            "Swaps": final_stats["swaps"],
            "Array Accesses": final_stats["array_accesses"],
            "Total Steps": len(steps)
        })
    
    return pd.DataFrame(results)

def main():
    st.title("🚀 Advanced Sorting Algorithm Visualizer")
    st.markdown("*Featuring advanced algorithms and performance analysis*")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    
    # Array configuration
    array_size = st.sidebar.slider("Array Size", min_value=10, max_value=100, value=30)
    
    array_type = st.sidebar.selectbox(
        "Array Type",
        ["random", "nearly_sorted", "reverse_sorted", "sorted", "duplicates"]
    )
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Select Advanced Algorithm",
        ["Heap Sort", "Cocktail Sort", "Shell Sort"]
    )
    
    # Animation speed
    animation_speed = st.sidebar.slider("Animation Speed (seconds)", min_value=0.05, max_value=1.0, value=0.2, step=0.05)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🎯 Visualizer", "📊 Algorithm Comparison", "📚 Algorithm Details"])
    
    with tab1:
        # Initialize visualizer
        visualizer = AdvancedSortingVisualizer()
        
        # Generate new array button
        if st.sidebar.button("🔄 Generate New Array"):
            st.session_state.array = visualizer.generate_array(array_size, array_type)
            st.session_state.sorted = False
            st.session_state.sorting = False
        
        # Initialize array if not exists
        if 'array' not in st.session_state or len(st.session_state.array) != array_size:
            st.session_state.array = visualizer.generate_array(array_size, array_type)
            st.session_state.sorted = False
            st.session_state.sorting = False
        
        # Main visualization area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"📊 {algorithm} Visualization")
            chart_placeholder = st.empty()
            
            # Show initial array
            if not st.session_state.get('sorting', False):
                fig = create_advanced_chart(st.session_state.array, title=f"Initial Array ({algorithm})")
                chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Performance Metrics")
            comparisons_metric = st.empty()
            swaps_metric = st.empty()
            accesses_metric = st.empty()
            
            # Show initial metrics
            comparisons_metric.metric("Comparisons", 0)
            swaps_metric.metric("Swaps", 0)
            accesses_metric.metric("Array Accesses", 0)
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_sorting = st.button("▶️ Start Sorting", disabled=st.session_state.get('sorting', False))
        
        with col2:
            reset_array = st.button("🔄 Reset")
        
        with col3:
            st.write(f"**Status:** {'✅ Sorted' if st.session_state.get('sorted', False) else '❌ Unsorted'}")
        
        # Reset functionality
        if reset_array:
            st.session_state.array = visualizer.generate_array(array_size, array_type)
            st.session_state.sorted = False
            st.session_state.sorting = False
            st.rerun()
        
        # Start sorting animation
        if start_sorting:
            st.session_state.sorting = True
            visualizer.reset_counters()
            
            # Status and progress
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            # Get the appropriate sorting algorithm
            if algorithm == "Heap Sort":
                sort_generator = visualizer.heapsort(st.session_state.array)
            elif algorithm == "Cocktail Sort":
                sort_generator = visualizer.cocktail_sort(st.session_state.array)
            elif algorithm == "Shell Sort":
                sort_generator = visualizer.shell_sort(st.session_state.array)
            
            # Animate sorting steps
            steps = list(sort_generator)
            total_steps = len(steps)
            
            for i, (current_array, idx1, idx2, description, stats) in enumerate(steps):
                # Update progress
                progress = (i + 1) / total_steps
                progress_bar.progress(progress)
                
                # Update status
                status_placeholder.write(f"**Step {i + 1}/{total_steps}:** {description}")
                
                # Update metrics
                comparisons_metric.metric("Comparisons", stats["comparisons"])
                swaps_metric.metric("Swaps", stats["swaps"])
                accesses_metric.metric("Array Accesses", stats["array_accesses"])
                
                # Update visualization
                fig = create_advanced_chart(
                    current_array,
                    highlight_indices=(idx1, idx2),
                    title=f"{algorithm} - Step {i + 1}",
                    stats=stats
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Animation delay
                time.sleep(animation_speed)
            
            # Update session state
            st.session_state.array = steps[-1][0]
            st.session_state.sorted = True
            st.session_state.sorting = False
            
            status_placeholder.success("🎉 Sorting completed successfully!")
            progress_bar.progress(1.0)
    
    with tab2:
        st.subheader("📊 Algorithm Performance Comparison")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            comparison_size = st.slider("Comparison Array Size", min_value=10, max_value=50, value=25)
            comparison_type = st.selectbox(
                "Comparison Array Type",
                ["random", "nearly_sorted", "reverse_sorted", "sorted", "duplicates"],
                key="comparison_type"
            )
        
        with col2:
            algorithms_to_compare = st.multiselect(
                "Select Algorithms to Compare",
                ["Heap Sort", "Cocktail Sort", "Shell Sort"],
                default=["Heap Sort", "Shell Sort"]
            )
        
        if st.button("🔍 Run Comparison") and algorithms_to_compare:
            with st.spinner("Running algorithm comparison..."):
                # Generate test array
                test_visualizer = AdvancedSortingVisualizer()
                test_array = test_visualizer.generate_array(comparison_size, comparison_type)
                
                # Compare algorithms
                comparison_df = compare_algorithms(test_array, algorithms_to_compare)
                
                # Display results
                st.subheader("📈 Comparison Results")
                st.dataframe(comparison_df, use_container_width=True)
                
                # Create comparison charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_comparisons = px.bar(
                        comparison_df, 
                        x="Algorithm", 
                        y="Comparisons",
                        title="Number of Comparisons",
                        color="Algorithm"
                    )
                    st.plotly_chart(fig_comparisons, use_container_width=True)
                
                with col2:
                    fig_swaps = px.bar(
                        comparison_df, 
                        x="Algorithm", 
                        y="Swaps",
                        title="Number of Swaps",
                        color="Algorithm"
                    )
                    st.plotly_chart(fig_swaps, use_container_width=True)
    
    with tab3:
        st.subheader("📚 Advanced Algorithm Details")
        
        algorithm_details = {
            "Heap Sort": {
                "Time Complexity": "O(n log n)",
                "Space Complexity": "O(1)",
                "Stability": "Not Stable",
                "Description": "Uses a binary heap data structure. First builds a max-heap, then repeatedly extracts the maximum element.",
                "Best Case": "O(n log n)",
                "Average Case": "O(n log n)",
                "Worst Case": "O(n log n)",
                "Advantages": ["Guaranteed O(n log n) performance", "In-place sorting", "Not affected by input distribution"],
                "Disadvantages": ["Not stable", "More complex than simpler algorithms", "Poor cache performance"]
            },
            "Cocktail Sort": {
                "Time Complexity": "O(n²)",
                "Space Complexity": "O(1)",
                "Stability": "Stable",
                "Description": "A variation of bubble sort that sorts in both directions. Also known as bidirectional bubble sort.",
                "Best Case": "O(n)",
                "Average Case": "O(n²)",
                "Worst Case": "O(n²)",
                "Advantages": ["Stable sorting", "Detects sorted arrays early", "Simple implementation"],
                "Disadvantages": ["Poor performance on large datasets", "More comparisons than bubble sort", "Still O(n²) complexity"]
            },
            "Shell Sort": {
                "Time Complexity": "O(n^(3/2))",
                "Space Complexity": "O(1)",
                "Stability": "Not Stable",
                "Description": "Generalization of insertion sort. Uses gap sequences to sort elements far apart first.",
                "Best Case": "O(n log n)",
                "Average Case": "O(n^(3/2))",
                "Worst Case": "O(n²)",
                "Advantages": ["Better than insertion sort", "In-place sorting", "Adaptive to input"],
                "Disadvantages": ["Complex gap sequence selection", "Not stable", "Performance depends on gap sequence"]
            }
        }
        
        selected_algo = st.selectbox("Select Algorithm for Details", list(algorithm_details.keys()))
        details = algorithm_details[selected_algo]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⏱️ Complexity Analysis")
            st.metric("Time Complexity", details["Time Complexity"])
            st.metric("Space Complexity", details["Space Complexity"])
            st.metric("Stability", details["Stability"])
            
            st.subheader("📊 Performance Cases")
            st.write(f"**Best Case:** {details['Best Case']}")
            st.write(f"**Average Case:** {details['Average Case']}")
            st.write(f"**Worst Case:** {details['Worst Case']}")
        
        with col2:
            st.subheader("📝 Description")
            st.write(details["Description"])
            
            st.subheader("✅ Advantages")
            for advantage in details["Advantages"]:
                st.write(f"• {advantage}")
            
            st.subheader("❌ Disadvantages")
            for disadvantage in details["Disadvantages"]:
                st.write(f"• {disadvantage}")

if __name__ == "__main__":
    main()
