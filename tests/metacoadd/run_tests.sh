#!/bin/bash

# This script has been generated with AI assistance.

set -e

# Configuration
TEST_DIR="tests/metacoadd"

# Start total timer
TOTAL_START=$(date +%s)

# Clean up old coverage data
echo "Cleaning up old coverage data..."
rm -f .coverage .coverage.*
rm -rf htmlcov

# Find all test files
TEST_FILES=$(find "$TEST_DIR" -name "test_*.py" -type f | sort)
TOTAL=$(echo "$TEST_FILES" | wc -l)
CURRENT=0

echo "Found $TOTAL test files"
echo ""

# Run each test file separately, appending to the same .coverage file
for test_file in $TEST_FILES; do
    CURRENT=$((CURRENT + 1))

    echo "=========================================="
    echo "[$CURRENT/$TOTAL] Running: $test_file"
    echo "=========================================="

    # Start timer for this test
    START=$(date +%s)

    if [ "$CURRENT" -eq 1 ]; then
        # First run: create a new coverage file
        COVERAGE_MODE=1 coverage run \
            --source=./src \
            -m pytest -sv "$test_file"
    else
        # Subsequent runs: append to the existing coverage file
        COVERAGE_MODE=1 coverage run \
            --append \
            --source=./src \
            -m pytest -sv "$test_file"
    fi

    # Calculate elapsed time
    END=$(date +%s)
    ELAPSED=$((END - START))

    echo "    Elapsed time: ${ELAPSED}s"
    echo ""
done

# Generate reports
echo "=========================================="
echo "Generating coverage reports..."
echo "=========================================="

coverage report -m
coverage html
coverage xml

# Calculate total elapsed time
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

# Convert to minutes and seconds if > 60 seconds
if [ "$TOTAL_ELAPSED" -ge 60 ]; then
    MINUTES=$((TOTAL_ELAPSED / 60))
    SECONDS=$((TOTAL_ELAPSED % 60))
    TIME_STR="${MINUTES}m ${SECONDS}s"
else
    TIME_STR="${TOTAL_ELAPSED}s"
fi

echo ""
echo "=========================================="
echo "Done! Coverage report at htmlcov/index.html"
echo "    Total time: $TIME_STR"
echo "=========================================="