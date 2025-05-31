import argparse
import logging
from typing import List, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_file_lines(filepath: str) -> List[str]:
    """
    Reads a file and returns the lines as a list.
    
    Args:
        filepath (str): The path to the file.
        
    Returns:
        list: A list of lines from the file.
    """
    lines = []
    try:
        with open(filepath, 'r', encoding='latin1') as file:
            lines = file.readlines()
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {e}")
    return lines

def calculate_intersection(set1: Set[str], set2: Set[str]) -> int:
    """
    Calculates the intersection count of two sets.
    
    Args:
        set1 (Set[str]): The first set.
        set2 (Set[str]): The second set.
        
    Returns:
        int: The count of intersecting elements.
    """
    return len(set1 & set2)

def check_model_effectiveness(model_output_path: str, test_dataset_path: str) -> None:
    """
    Check the effectiveness of the model by comparing the model output with the test dataset.
    
    Args:
        model_output_path (str): The path to the model output.
        test_dataset_path (str): The path to the test dataset.
        
    Returns:
        None
    """
    logging.info("Starting the effectiveness check of the model.")

    # Read model output
    model_outputs = read_file_lines(model_output_path)
    if not model_outputs:
        logging.warning(f"No data found in model output file: {model_output_path}")
        return

    # Read test dataset
    test_data = read_file_lines(test_dataset_path)
    if not test_data:
        logging.warning(f"No data found in test dataset file: {test_dataset_path}")
        return

    logging.info(f"Model output size: {len(model_outputs)}")
    logging.info(f"Test data size: {len(test_data)}")
    
    # Calculate intersection
    model_output_set = set(model_outputs)
    test_data_set = set(test_data)
    intersection_count = calculate_intersection(model_output_set, test_data_set)

    effectiveness_percentage = (intersection_count / len(test_data_set) * 100) if test_data_set else 0
    logging.info(f"Intersection: {intersection_count} matches, {effectiveness_percentage:.2f}%")
    print(f"Intersection: {intersection_count} matches, {effectiveness_percentage:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check model effectiveness by checking cover rate of model output on test dataset.")
    parser.add_argument("--model_output", type=str, required=True, help="The model output file.")
    parser.add_argument("--test_data", type=str, required=True, help="The test data file.")
    args = parser.parse_args()

    check_model_effectiveness(args.model_output, args.test_data)
