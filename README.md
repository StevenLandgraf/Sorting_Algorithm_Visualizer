# 🎯 Sorting Algorithm Visualizer

An interactive Streamlit application that visualizes various sorting algorithms with real-time animations and performance metrics.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### Basic Visualizer (`app.py`)
- **5 Classic Algorithms**: Bubble Sort, Selection Sort, Insertion Sort, Merge Sort, Quick Sort
- **Interactive Controls**: Adjustable array size, animation speed, and value ranges
- **Real-time Visualization**: Watch elements being compared and swapped in real-time
- **Algorithm Information**: Time/space complexity and descriptions for each algorithm
- **Array Statistics**: View min, max, sum, and other statistics

### Advanced Visualizer (`advanced_app.py`)
- **3 Advanced Algorithms**: Heap Sort, Cocktail Sort (Bidirectional Bubble Sort), Shell Sort
- **Performance Metrics**: Track comparisons, swaps, and array accesses in real-time
- **Algorithm Comparison**: Side-by-side performance analysis of multiple algorithms
- **Multiple Array Types**: Random, nearly sorted, reverse sorted, sorted, and duplicate arrays
- **Detailed Documentation**: Comprehensive algorithm details with complexity analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/StevenLandgraf/Sorting_Algorithm_Visualizer.git
   cd Sorting_Algorithm_Visualizer
   ```

2. **Create and activate virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Option 1: Using the Launcher Script
```bash
./run.sh
```
Choose between:
1. Basic Visualizer
2. Advanced Visualizer

#### Option 2: Direct Launch
**Basic Visualizer:**
```bash
streamlit run app.py
```

**Advanced Visualizer:**
```bash
streamlit run advanced_app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📊 Algorithms Included

### Basic Algorithms
| Algorithm | Time Complexity | Space Complexity | Stable | Description |
|-----------|----------------|------------------|--------|-------------|
| **Bubble Sort** | O(n²) | O(1) | ✅ | Compares adjacent elements and swaps them |
| **Selection Sort** | O(n²) | O(1) | ❌ | Finds minimum element and places it at beginning |
| **Insertion Sort** | O(n²) | O(1) | ✅ | Builds sorted array one element at a time |
| **Merge Sort** | O(n log n) | O(n) | ✅ | Divides array and merges sorted halves |
| **Quick Sort** | O(n log n) avg | O(log n) | ❌ | Partitions array around pivot element |

### Advanced Algorithms
| Algorithm | Time Complexity | Space Complexity | Stable | Description |
|-----------|----------------|------------------|--------|-------------|
| **Heap Sort** | O(n log n) | O(1) | ❌ | Uses binary heap data structure |
| **Cocktail Sort** | O(n²) | O(1) | ✅ | Bidirectional bubble sort |
| **Shell Sort** | O(n^1.5) | O(1) | ❌ | Generalization of insertion sort with gaps |

## 🎮 How to Use

### Basic Visualizer
1. **Configure Array**: Set size (5-50), minimum/maximum values
2. **Choose Algorithm**: Select from 5 classic sorting algorithms
3. **Adjust Speed**: Control animation speed (0.1-2.0 seconds per step)
4. **Generate Array**: Click "Generate New Array" for random data
5. **Start Sorting**: Click "Start Sorting" to begin visualization
6. **Monitor Progress**: Watch real-time sorting with highlighted comparisons

### Advanced Visualizer
1. **Select Algorithm**: Choose from 3 advanced algorithms
2. **Choose Array Type**: 
   - Random: Completely random values
   - Nearly Sorted: Mostly sorted with few swaps
   - Reverse Sorted: Descending order
   - Sorted: Already sorted (best case)
   - Duplicates: Many duplicate values
3. **Performance Analysis**: View real-time metrics (comparisons, swaps, array accesses)
4. **Algorithm Comparison**: Compare multiple algorithms side-by-side
5. **Detailed Information**: Access comprehensive algorithm documentation

## 📈 Performance Metrics

The advanced visualizer tracks:
- **Comparisons**: Number of element comparisons
- **Swaps**: Number of element exchanges
- **Array Accesses**: Total array read/write operations
- **Execution Time**: Time taken to complete sorting
- **Total Steps**: Number of visualization steps

## 🛠️ Project Structure

```
Sorting_Algorithm_Visualizer/
│
├── app.py                 # Basic visualizer with 5 classic algorithms
├── advanced_app.py        # Advanced visualizer with performance analysis
├── requirements.txt       # Python dependencies
├── run.sh                # Launcher script
├── README.md             # Project documentation
└── .venv/                # Virtual environment (created after setup)
```

## 🎯 Educational Value

This visualizer is perfect for:
- **Computer Science Students**: Understanding algorithm mechanics
- **Educators**: Teaching sorting algorithms interactively
- **Self-Learners**: Exploring algorithm performance differences
- **Interview Preparation**: Visualizing common algorithm questions

## 🔧 Customization

### Adding New Algorithms
1. Create a generator function in the `SortingVisualizer` class
2. Yield tuples of `(array_state, index1, index2, description)`
3. Add the algorithm to the selectbox options
4. Include algorithm information in the info dictionary

### Modifying Visualizations
- Colors can be customized in the `create_bar_chart()` function
- Chart types can be modified using Plotly's extensive options
- Animation speed and step granularity can be adjusted

## 📋 Requirements

```
streamlit>=1.28.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
plotly>=5.15.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Streamlit for the amazing web app framework
- Plotly for interactive visualizations
- The computer science community for algorithm development

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/StevenLandgraf/Sorting_Algorithm_Visualizer/issues) page
2. Create a new issue with detailed description
3. Contact the maintainer

---

**Happy Sorting! 🎉**