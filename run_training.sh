#!/bin/tcsh

# Activate virtual environment
source /export/qhu56/ML-dont-care/.venv/bin/activate.csh

# Change to src directory
cd /export/qhu56/ML-dont-care/src

# Run the draft.py script
echo "Starting training..."
python draft.py

echo "Training completed. Check results.txt for output."
