#!/bin/bash

# Sorting Algorithm Visualizer Launcher
echo "🎯 Sorting Algorithm Visualizer"
echo "================================"
echo ""
echo "Choose which version to run:"
echo "1. Basic Visualizer (Bubble, Selection, Insertion, Merge, Quick Sort)"
echo "2. Advanced Visualizer (Heap, Cocktail, Shell Sort with performance analysis)"
echo ""

read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        echo "Starting Basic Visualizer..."
        /Users/stevenl/Documents/GitHub/Sorting_Algorithm_Visualizer/.venv/bin/python -m streamlit run app.py
        ;;
    2)
        echo "Starting Advanced Visualizer..."
        /Users/stevenl/Documents/GitHub/Sorting_Algorithm_Visualizer/.venv/bin/python -m streamlit run advanced_app.py
        ;;
    *)
        echo "Invalid choice. Please run the script again and choose 1 or 2."
        ;;
esac
