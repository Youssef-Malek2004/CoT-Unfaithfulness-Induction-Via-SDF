#!/bin/bash
# ============================================================================
# Cobalt AI Dataset Generator - Quick Start Script
# ============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           COBALT AI DATASET GENERATOR                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Parse arguments
TOTAL=${1:-40000}
WORKERS=${2:-20}
BATCH_SIZE=${3:-10}

echo ""
echo "Configuration:"
echo "  📊 Total documents: $TOTAL"
echo "  ⚡ Concurrent workers: $WORKERS"
echo "  📦 Batch size: $BATCH_SIZE docs/call"
echo ""

# Calculate estimates
TOTAL_CALLS=$((TOTAL / BATCH_SIZE))
# Assuming ~2 seconds per call with concurrency
EST_SECONDS=$((TOTAL_CALLS * 2 / WORKERS))
EST_HOURS=$((EST_SECONDS / 3600))
EST_MINS=$(((EST_SECONDS % 3600) / 60))

echo "  📞 Estimated API calls: $TOTAL_CALLS"
echo "  ⏱️  Estimated time: ${EST_HOURS}h ${EST_MINS}m"
echo ""

read -p "Press Enter to start generation (Ctrl+C to cancel)..."

# Run generator
python batch_generator.py \
    --total "$TOTAL" \
    --workers "$WORKERS" \
    --batch-size "$BATCH_SIZE" \
    --resume

echo ""
echo "✅ Generation complete!"

