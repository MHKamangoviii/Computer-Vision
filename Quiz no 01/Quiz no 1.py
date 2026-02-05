import cv2
import numpy as np
import os
import sys

def split_payment_slips_traditional(image_path, output_folder="split_slips"):
    """
    Split payment slips using traditional computer vision techniques (OpenCV).
    
    Args:
        image_path (str): Path to the input image
        output_folder (str): Folder to save the split images
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        print(f"Please make sure the file exists and is a valid image.")
        return 0
    
    # Store original image for reference
    original = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to clean up the image
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area to get potential receipt regions
    min_area = 5000  # Minimum area for a receipt
    max_area = image.shape[0] * image.shape[1] * 0.8  # Maximum area (80% of image)
    
    receipt_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (receipts are typically rectangular)
            aspect_ratio = w / h
            if 0.3 < aspect_ratio < 4.0:  # Reasonable aspect ratio for receipts
                receipt_contours.append((contour, area, (x, y, w, h)))
    
    # Sort contours by y-coordinate (top to bottom)
    receipt_contours.sort(key=lambda x: x[2][1])
    
    # Extract and save each receipt
    extracted_count = 0
    for i, (contour, area, (x, y, w, h)) in enumerate(receipt_contours):
        # Add padding around the detected region
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # Extract the receipt region
        receipt_img = original[y1:y2, x1:x2]
        
        # Only save if the extracted image has reasonable dimensions
        if receipt_img.shape[0] > 50 and receipt_img.shape[1] > 50:
            extracted_count += 1
            output_path = os.path.join(output_folder, f"receipt_{extracted_count}.png")
            cv2.imwrite(output_path, receipt_img)
            print(f"✓ Saved receipt {extracted_count} to {output_path} (size: {receipt_img.shape[1]}x{receipt_img.shape[0]})")
            
            # Optional: Draw rectangles on original image for visualization
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Receipt {extracted_count}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the annotated image with rectangles
    annotated_path = os.path.join(output_folder, "annotated_original.png")
    cv2.imwrite(annotated_path, image)
    
    print(f"\nTraditional Method Results:")
    print(f"Total receipts found: {extracted_count}")
    print(f"Annotated image saved to: {annotated_path}")
    
    return extracted_count

def split_payment_slips_enhanced(image_path, output_folder="enhanced_split"):
    """
    Enhanced method using edge detection and text detection for better results.
    
    Args:
        image_path (str): Path to the input image
        output_folder (str): Folder to save the split images
    """
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return 0
    
    original = image.copy()
    height, width = image.shape[:2]
    
    print(f"Processing image of size: {width}x{height}")
    
    # Method 1: Edge detection approach
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours on edges
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get potential receipt regions from edges
    edge_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Filter by size
        if area > 10000 and area < height * width * 0.7:
            # Check if it's reasonably rectangular
            if 0.3 < w/h < 4.0:
                edge_boxes.append((x, y, w, h))
    
    # Method 2: Text-based detection (receipts have lots of text)
    # Use morphological gradient to find text regions
    gradient_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, gradient_kernel)
    
    # Threshold the gradient
    _, text_mask = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)
    
    # Close gaps in text regions
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    closed_text = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, text_kernel)
    
    # Find contours in text regions
    text_contours, _ = cv2.findContours(closed_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_boxes = []
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Filter text regions that could be receipts
        if area > 15000 and area < height * width * 0.6:
            # Receipts have more height than width typically
            if h > 100 and w > 150:
                text_boxes.append((x, y, w, h))
    
    # Combine boxes from both methods
    all_boxes = edge_boxes + text_boxes
    
    # Merge overlapping boxes
    filtered_boxes = []
    for box in all_boxes:
        x, y, w, h = box
        overlap = False
        
        # Check for significant overlap with existing boxes
        for i, (fx, fy, fw, fh) in enumerate(filtered_boxes):
            # Calculate overlap area
            x_left = max(x, fx)
            y_top = max(y, fy)
            x_right = min(x + w, fx + fw)
            y_bottom = min(y + h, fy + fh)
            
            if x_right > x_left and y_bottom > y_top:
                overlap_area = (x_right - x_left) * (y_bottom - y_top)
                box_area = w * h
                other_area = fw * fh
                
                # If significant overlap, merge boxes
                if overlap_area > 0.5 * min(box_area, other_area):
                    # Merge boxes
                    new_x = min(x, fx)
                    new_y = min(y, fy)
                    new_w = max(x + w, fx + fw) - new_x
                    new_h = max(y + h, fy + fh) - new_y
                    
                    filtered_boxes[i] = (new_x, new_y, new_w, new_h)
                    overlap = True
                    break
        
        if not overlap:
            filtered_boxes.append(box)
    
    # Sort by y-coordinate (top to bottom)
    filtered_boxes.sort(key=lambda b: b[1])
    
    # Extract and save receipts
    extracted_count = 0
    for i, (x, y, w, h) in enumerate(filtered_boxes):
        # Add generous padding
        padding = 25
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)
        
        receipt = original[y1:y2, x1:x2]
        
        # Only save if reasonable size
        if receipt.shape[0] > 100 and receipt.shape[1] > 100:
            extracted_count += 1
            output_path = os.path.join(output_folder, f"receipt_{extracted_count}.png")
            cv2.imwrite(output_path, receipt)
            print(f"✓ Saved receipt {extracted_count} to {output_path} (size: {receipt.shape[1]}x{receipt.shape[0]})")
            
            # Draw on annotated image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, f"Receipt {extracted_count}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save the annotated image
    annotated_path = os.path.join(output_folder, "annotated_original.png")
    cv2.imwrite(annotated_path, image)
    
    print(f"\nEnhanced Method Results:")
    print(f"Total receipts found: {extracted_count}")
    print(f"Annotated image saved to: {annotated_path}")
    
    return extracted_count

def main():
    """
    Main function to run the receipt splitting program.
    """
    
    # Define the image file name
    image_file = "input.png"
    
    # Check if the image file exists
    if not os.path.exists(image_file):
        print(f"Error: Image file '{image_file}' not found in the current directory!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        
        # Ask user if they want to try with a different filename
        alt_file = input("\nEnter the correct image filename (or press Enter to exit): ").strip()
        if alt_file:
            image_file = alt_file
        else:
            sys.exit(1)
    
    # Check again with the new filename
    if not os.path.exists(image_file):
        print(f"Error: Image file '{image_file}' not found!")
        sys.exit(1)
    
    print("=" * 60)
    print("RECEIPT SPLITTER TOOL")
    print("=" * 60)
    print(f"Processing image: {image_file}")
    print(f"Image size: {os.path.getsize(image_file)} bytes")
    print("=" * 60)
    
    # Run traditional method
    print("\n" + "=" * 60)
    print("METHOD 1: Traditional Contour Detection")
    print("=" * 60)
    count1 = split_payment_slips_traditional(image_file, "output_traditional")
    
    # Run enhanced method
    print("\n" + "=" * 60)
    print("METHOD 2: Enhanced Edge Detection")
    print("=" * 60)
    count2 = split_payment_slips_enhanced(image_file, "output_enhanced")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Traditional Method: {count1} receipts found")
    print(f"Enhanced Method: {count2} receipts found")
    
    # Determine which method worked better
    if count1 == 0 and count2 == 0:
        print("\n⚠️  No receipts were detected by either method!")
        print("Possible reasons:")
        print("  1. The image quality might be poor")
        print("  2. Receipts might not have clear borders")
        print("  3. Try scanning the receipts with better contrast")
    elif count1 > count2:
        print(f"\n✅ Traditional method worked better ({count1} vs {count2} receipts)")
        print(f"   Check the 'output_traditional' folder for results")
    elif count2 > count1:
        print(f"\n✅ Enhanced method worked better ({count2} vs {count1} receipts)")
        print(f"   Check the 'output_enhanced' folder for results")
    else:
        print(f"\n✅ Both methods found the same number of receipts: {count1}")
        print(f"   Compare results in both output folders")
    
    print("\n" + "=" * 60)
    print("OUTPUT FOLDERS:")
    print("=" * 60)
    print("• output_traditional/ - Traditional contour detection results")
    print("• output_enhanced/   - Enhanced edge detection results")
    print("\nEach folder contains:")
    print("  - receipt_1.png, receipt_2.png, etc. - Individual receipts")
    print("  - annotated_original.png - Original image with detection boxes")
    print("=" * 60)
    
    # Open the output folders (Windows)
    try:
        if os.path.exists("output_traditional"):
            os.startfile("output_traditional")
        if os.path.exists("output_enhanced"):
            os.startfile("output_enhanced")
    except:
        pass  # If os.startfile is not available (non-Windows), just continue

if __name__ == "__main__":
    # Check if OpenCV is installed
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Error: Required packages are not installed!")
        print("Please install OpenCV and NumPy:")
        print("  pip install opencv-python numpy")
        sys.exit(1)
    
    # Run the main program
    main()