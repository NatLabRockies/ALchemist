"""
Example script demonstrating complete ALchemist API workflow.

Run the API server first:
    python api/main.py

Then run this script:
    python api/example_client.py
"""

import requests
import time

# Configuration
BASE_URL = "http://localhost:8000/api/v1"


def main():
    print("=" * 60)
    print("ALchemist API Example Workflow")
    print("=" * 60)
    
    # 1. Create session
    print("\n1. Creating optimization session...")
    response = requests.post(f"{BASE_URL}/sessions", json={})
    response.raise_for_status()
    session_id = response.json()["session_id"]
    print(f"   ✓ Session created: {session_id}")
    
    # 2. Define search space
    print("\n2. Defining search space...")
    
    # Temperature variable
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/variables",
        json={
            "name": "temperature",
            "type": "continuous",
            "bounds": [100, 500],
            "unit": "°C",
            "description": "Reaction temperature"
        }
    )
    response.raise_for_status()
    print(f"   ✓ Added variable: temperature [100-500°C]")
    
    # Pressure variable
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/variables",
        json={
            "name": "pressure",
            "type": "continuous",
            "bounds": [1, 10],
            "unit": "bar",
            "description": "Reaction pressure"
        }
    )
    response.raise_for_status()
    print(f"   ✓ Added variable: pressure [1-10 bar]")
    
    # Catalyst type (categorical)
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/variables",
        json={
            "name": "catalyst",
            "type": "categorical",
            "categories": ["Pt", "Pd", "Rh"],
            "description": "Catalyst type"
        }
    )
    response.raise_for_status()
    print(f"   ✓ Added variable: catalyst [Pt, Pd, Rh]")
    
    # 3. Add initial experimental data
    print("\n3. Adding experimental data...")
    
    experiments = [
        {"inputs": {"temperature": 200, "pressure": 3, "catalyst": "Pt"}, "output": 0.65},
        {"inputs": {"temperature": 250, "pressure": 5, "catalyst": "Pd"}, "output": 0.85},
        {"inputs": {"temperature": 300, "pressure": 7, "catalyst": "Pt"}, "output": 0.92},
        {"inputs": {"temperature": 350, "pressure": 4, "catalyst": "Rh"}, "output": 0.78},
        {"inputs": {"temperature": 400, "pressure": 6, "catalyst": "Pd"}, "output": 0.88},
        {"inputs": {"temperature": 450, "pressure": 8, "catalyst": "Pt"}, "output": 0.81},
        {"inputs": {"temperature": 275, "pressure": 5.5, "catalyst": "Rh"}, "output": 0.79},
        {"inputs": {"temperature": 325, "pressure": 6.5, "catalyst": "Pd"}, "output": 0.91},
    ]
    
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/experiments/batch",
        json={"experiments": experiments}
    )
    response.raise_for_status()
    batch_result = response.json()
    n_added = batch_result.get("n_added", len(experiments))
    total_experiments = batch_result.get("n_experiments")
    if total_experiments is not None and total_experiments >= n_added:
        print(f"   ✓ Added {n_added} experiments (total: {total_experiments})")
    else:
        print(f"   ✓ Added {n_added} experiments")
    
    # Get data summary
    response = requests.get(f"{BASE_URL}/sessions/{session_id}/experiments/summary")
    response.raise_for_status()
    summary = response.json()
    print(f"   ✓ Output range: [{summary['output_range'][0]:.2f}, {summary['output_range'][1]:.2f}]")
    
    # 4. Train surrogate model
    print("\n4. Training surrogate model...")
    
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/model/train",
        json={
            "backend": "sklearn",
            "kernel": "rbf",
            "output_transform": "standardize"
        }
    )
    response.raise_for_status()
    train_results = response.json()
    print(f"   ✓ Model trained: {train_results['backend']}")
    print(f"   ✓ Kernel: {train_results['kernel']}")
    if train_results.get("metrics"):
        print(f"   ✓ R² Score: {train_results['metrics'].get('r2_score', 'N/A'):.3f}")
    
    # 5. Get next experiment suggestions
    print("\n5. Getting next experiment suggestions...")
    
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/acquisition/suggest",
        json={
            "strategy": "EI",
            "goal": "maximize",
            "n_suggestions": 3
        }
    )
    response.raise_for_status()
    suggestions = response.json()["suggestions"]
    
    print(f"   ✓ Generated {len(suggestions)} suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"      {i}. T={suggestion['temperature']:.1f}°C, "
              f"P={suggestion['pressure']:.1f}bar, "
              f"Cat={suggestion['catalyst']}")
    
    # 6. Make predictions at specific points
    print("\n6. Making predictions...")
    
    test_points = [
        {"temperature": 275, "pressure": 5.5, "catalyst": "Pt"},
        {"temperature": 325, "pressure": 6.5, "catalyst": "Pd"},
        {"temperature": 375, "pressure": 7.5, "catalyst": "Rh"},
    ]
    
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/model/predict",
        json={"inputs": test_points}
    )
    response.raise_for_status()
    predictions = response.json()["predictions"]
    
    print(f"   ✓ Predictions:")
    for pred in predictions:
        inputs = pred["inputs"]
        print(f"      T={inputs['temperature']:.0f}°C, P={inputs['pressure']:.1f}bar, "
              f"Cat={inputs['catalyst']}: "
              f"y={pred['prediction']:.3f} ± {pred['uncertainty']:.3f}")
    
    # 7. Session info
    print("\n7. Session information...")
    response = requests.get(f"{BASE_URL}/sessions/{session_id}")
    response.raise_for_status()
    session_info = response.json()
    print(f"   ✓ Session ID: {session_info['session_id']}")
    print(f"   ✓ Created: {session_info['created_at']}")
    
    print("\n" + "=" * 60)
    print("Workflow complete!")
    print("=" * 60)
    print(f"\nSession ID: {session_id}")
    print("View API docs: http://localhost:8000/api/docs")
    print("\nTo clean up, delete the session:")
    print(f"  requests.delete('{BASE_URL}/sessions/{session_id}')")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server.")
        print("Make sure the server is running:")
        print("  python api/main.py")
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
