# Smart Route Optimizer

A sophisticated route optimization system built with Python, utilizing deep reinforcement learning to efficiently plan delivery routes while considering multiple constraints and objectives.

## Features

**Core Capabilities**
- Multi-vehicle route optimization
- Time window compliance
- Capacity constraints handling
- Dynamic vehicle selection
- Real-time visualization

**Vehicle Support**
- Three-wheeler (3W) with 5km radius
- Electric four-wheeler (4W-EV) with 8km radius  
- Standard four-wheeler (4W) with extended range

## Technical Architecture

**Neural Network Components**
- Deep Q-Network (DQN) implementation
- Experience replay buffer
- Target network for stable training
- Epsilon-greedy exploration strategy

**Key Metrics**
- Minimum Spanning Tree (MST) distance calculation
- Time utilization optimization
- Coverage utilization tracking
- Capacity utilization monitoring

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-route-optimizer.git

# Install required packages
pip install -r requirements.txt
```

## Usage

```python
# Import the main optimization function
from main import main1

# Load your shipment data as a pandas DataFrame
# Required columns: Shipment ID, Latitude, Longitude, Delivery Timeslot
trip_df = main1(shipments_df)
```

## Output

The system generates:
- Detailed trip plans in Excel format
- Interactive map visualization using Folium
- Trip quality metrics and statistics

## Dependencies

- PyTorch
- Pandas
- NetworkX
- Folium
- XlsxWriter
- Streamlit

## Web Interface

The project includes a Streamlit-based web interface that allows:
- File upload (CSV/Excel)
- Interactive map visualization
- Download of optimized route plans
