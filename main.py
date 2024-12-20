import numpy as np

def generate_dataset(num_points=4000, num_dimensions=5, filename='data.txt'):
    # Generate random data points
    data = np.random.rand(num_points, num_dimensions) * 100  # Scale points to 0-100
    
    # Prepare the dataset in the desired format
    with open(filename, 'w') as f:
        f.write(f"{num_points} {num_dimensions}\n")
        for point in data:
            f.write(" ".join(map(str, point)) + "\n")

if __name__ == "__main__":
    generate_dataset()
    print("Dataset generated and saved to 'data.txt'.")
