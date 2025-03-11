import cv2
import numpy as np
import torch
import os
import time
from lightglue import LightGlue, SuperPoint # type: ignore
from lightglue.utils import match_pair, load_image, rbd # type: ignore

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Parameters
    min_object_area = 1000  # Minimum area in pixels to consider a valid object
    stability_threshold = 100  # Maximum movement in pixels to consider object stable
    stability_frames = 200  # Number of frames object must be stable to consider it stopped
    
    # Step 1: Capture reference frame
    reference_frame = capture_reference_frame(cap)
    
    # Step 2: Monitor for movement and capture moving object
    isolated_object, final_frame = detect_and_track_object(
        cap, reference_frame, min_object_area, stability_threshold, stability_frames
    )
    
    # Step 3: Use LightGlue to find the object in the final frame
    highlighted_frame = find_object_with_lightglue(isolated_object, final_frame)
    
    # Create a window to display all three images
    result = np.zeros((max(reference_frame.shape[0], isolated_object.shape[0], highlighted_frame.shape[0]), 
                      reference_frame.shape[1] + isolated_object.shape[1] + highlighted_frame.shape[1], 
                      3), dtype=np.uint8)
    
    # Place the three images side by side
    result[:reference_frame.shape[0], :reference_frame.shape[1]] = reference_frame
    result[:isolated_object.shape[0], reference_frame.shape[1]:reference_frame.shape[1] + isolated_object.shape[1]] = isolated_object
    result[:highlighted_frame.shape[0], reference_frame.shape[1] + isolated_object.shape[1]:] = highlighted_frame
    
    # Show the result
    cv2.imshow("Object Tracking Results", result)
    cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()

def capture_reference_frame(cap):
    """Capture a reference frame from the camera when user presses 'c'"""
    print("Press 'c' to capture reference frame...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera.")
            return None
        
        # Display the current frame
        cv2.imshow("Press 'c' to capture reference frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            cv2.destroyAllWindows()
            print("Reference frame captured! Now watching for movement...")
            return frame.copy()
        elif key == 27:  # ESC key
            print("Exiting...")
            cv2.destroyAllWindows()
            return None


def detect_and_track_object(cap, reference_frame, min_area, stability_threshold, stability_frames_threshold):
    """Monitor live feed for movement, detect object, and wait for it to stabilize"""
    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)
    
    # Train the background subtractor with the reference frame
    bg_mask = bg_subtractor.apply(reference_frame)
    # Apply it again to get a better model
    for _ in range(5):
        bg_mask = bg_subtractor.apply(reference_frame, learningRate=0.1)
    
    # Variables for motion detection
    motion_detected = False
    stability_count = 0
    best_isolated_object = None
    largest_contour_area = 0
    object_frame = None
    
    # Variables for tracking stability
    prev_center = None
    prev_contour = None
    
    print("Watching for movement... (Press ESC to exit)")
    
    while True:
        ret, current_frame = cap.read()
        if not ret:
            print("Error reading from camera.")
            return None, None
        
        # Apply background subtraction to get foreground mask
        fg_mask = bg_subtractor.apply(current_frame, learningRate=0)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((10, 10), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Threshold to convert to binary mask (removing shadows)
        _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for significant motion
        significant_motion = False
        largest_contour = None
        max_area = 0
        current_center = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                significant_motion = True
                if area > max_area:
                    max_area = area
                    largest_contour = contour
                    # Calculate center of contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        current_center = (cx, cy)
        
        # Display frame with contours
        display_frame = current_frame.copy()
        
        if significant_motion and largest_contour is not None:
            # Draw the largest contour
            cv2.drawContours(display_frame, [largest_contour], -1, (0, 255, 0), 2)
            
            # Create isolated object view (black background with colored object)
            isolated = np.zeros_like(current_frame)
            mask = np.zeros_like(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY))
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            isolated[mask == 255] = current_frame[mask == 255]
            
            # Show isolated object in a separate window
            cv2.imshow("Current Isolated Object", isolated)
            
            # Keep track of the best isolated object (largest area)
            if max_area > largest_contour_area:
                largest_contour_area = max_area
                best_isolated_object = isolated.copy()
                object_frame = current_frame.copy()
            
            if not motion_detected:
                motion_detected = True
                
            # Check if object is stable
            if prev_center is not None and current_center is not None:
                # Calculate movement distance
                distance = np.sqrt((prev_center[0] - current_center[0])**2 + 
                                  (prev_center[1] - current_center[1])**2)
                
                # Calculate contour similarity if applicable
                contour_change = 100  # Default to a large value
                if prev_contour is not None:
                    # Compare contour areas
                    area_diff = abs(cv2.contourArea(prev_contour) - cv2.contourArea(largest_contour))
                    area_percent = area_diff / max(cv2.contourArea(prev_contour), 1) * 100
                    contour_change = area_percent
                
                # Check if object is stable (minimal movement and contour change)
                if distance < stability_threshold and contour_change < 20:
                    stability_count += 1

                    if stability_count >= stability_frames_threshold:
                        print("Object has stabilized!")
                        final_frame = current_frame.copy()
                        cv2.destroyWindow("Current Isolated Object")
                        return best_isolated_object, final_frame
                else:
                    stability_count = max(0, stability_count - 1)  # Decay stability count
            
            # Update previous center and contour for next frame comparison
            prev_center = current_center
            prev_contour = largest_contour
        else:
            # Reset stability tracking if no object detected
            stability_count = 0
            prev_center = None
            prev_contour = None
        
        # Show status and foreground mask

        cv2.imshow("Motion Detection", display_frame)
        cv2.imshow("Foreground Mask", fg_mask)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return None, None
        elif key == ord('c'):  # Allow manual capture
            print("Manual capture triggered!")
            if best_isolated_object is not None:
                cv2.destroyWindow("Current Isolated Object")
                return best_isolated_object, current_frame.copy()

def find_object_with_lightglue(isolated_object, final_frame):
    """Use LightGlue to find the object in the final frame"""
    print("Processing with LightGlue to locate the object in the final frame...")
    
    # Save the images temporarily for LightGlue processing
    cv2.imwrite('isolated_object.jpg', isolated_object)
    cv2.imwrite('final_frame.jpg', final_frame)
    
    try:
        # Use LightGlue to find the object
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize feature extractors and matcher
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        matcher = LightGlue(features='superpoint').eval().to(device)
        
        # Load images for LightGlue
        image0 = load_image('isolated_object.jpg', device=device)
        image1 = load_image('final_frame.jpg', device=device)
        
        # Extract features
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)
        
        # Match features
        matches = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches = [rbd(x) for x in [feats0, feats1, matches]]  # Remove batch dimension
        
        # Get keypoints and matches
        kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches['matches']
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        
        # Convert tensors to numpy arrays for OpenCV
        m_kpts0_np = m_kpts0.cpu().numpy()  
        m_kpts1_np = m_kpts1.cpu().numpy()  
        
        # Draw matches on the final frame
        highlighted_frame = final_frame.copy()
        
        # Draw keypoints and matches
        if len(m_kpts1_np) > 0:
            # Draw matched keypoints
            for pt in m_kpts1_np:
                pt_int = tuple(map(int, pt))
                cv2.circle(highlighted_frame, pt_int, 3, (0, 0, 255), -1)
            
            # Find the bounding box of matched keypoints
            x_min = int(np.min(m_kpts1_np[:, 0]))
            y_min = int(np.min(m_kpts1_np[:, 1]))
            x_max = int(np.max(m_kpts1_np[:, 0]))
            y_max = int(np.max(m_kpts1_np[:, 1]))
            
            # Draw bounding box around the object
            cv2.rectangle(highlighted_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            cv2.imwrite("cropped_image.jpg", highlighted_frame[y_min:y_max, x_min:x_max])            
            
            # Add label for matched points count
            cv2.putText(highlighted_frame, f"Matches: {len(m_kpts1_np)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(highlighted_frame, "No matches found", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return highlighted_frame
    
    except Exception as e:
        print(f"Error in LightGlue processing: {e}")
        highlighted_frame = final_frame.copy()
        cv2.putText(highlighted_frame, "LightGlue processing failed", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return highlighted_frame
    
    finally:
        # Clean up temporary files
        for file in ['isolated_object.jpg', 'final_frame.jpg']:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    main()