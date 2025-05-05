import pickle
import sys

def inspect_pkl(filepath):
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        print(f"Successfully loaded {filepath}")
        print("Type of loaded data:", type(data))

        if hasattr(data, 'keys'):
            print("Keys:", data.keys())

        else:
            print("Could not determine structure (not dict-like)")

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except pickle.UnpicklingError:
        print(f"Error: Could not unpickle data from {filepath}. It might be corrupted or not a valid pickle file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_sbm_pkl.py <path_to_pkl_file>")
        sys.exit(1)

    pkl_filepath = sys.argv[1]
    inspect_pkl(pkl_filepath)