import cv2
import numpy as np
import torch
import os
import time
from lightglue import LightGlue, SuperPoint # type: ignore
from lightglue.utils import match_pair, load_image, rbd # type: ignore

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Parameters
    min_object_area = 1000  # Minimum area to consider valid
    stability_threshold = 200  # Max movement in pixels to consider stable
    stability_frames = 75 # Number of frames object must be stable
    
    # Step 1: Capture reference frame
    reference_frame = capture_reference_frame(cap)
    if reference_frame is None:
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Step 2: Monitor for movement and capture moving object
    isolated_object, final_frame = detect_and_track_object(
        cap, reference_frame, min_object_area, stability_threshold, stability_frames
    )
    
    if isolated_object is None or final_frame is None:
        print("Object detection failed.")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Step 3: Use LightGlue to find the object in the final frame
    highlighted_frame = find_object_with_lightglue(isolated_object, final_frame)
    
    # Create a window to display two images side by side
    result = np.zeros((max(isolated_object.shape[0], highlighted_frame.shape[0]), 
                      isolated_object.shape[1] + highlighted_frame.shape[1], 
                      3), dtype=np.uint8)
    
    # Place the two images side by side
    result[:isolated_object.shape[0], :isolated_object.shape[1]] = isolated_object
    result[:highlighted_frame.shape[0], isolated_object.shape[1]:] = highlighted_frame
    
    # Show the result
    cv2.imshow("Object Detection Results", result)
    cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()

def capture_reference_frame(cap):
    """Capture a reference frame from the camera when user presses 'c'"""
    print("Press 'c' to capture reference frame or ESC to exit...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera.")
            return None
        
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

@torch.no_grad()
def detect_and_track_object(cap, reference_frame, min_area, stability_threshold, stability_frames_threshold):
    """Monitor live feed for movement, detect object, and wait for it to stabilize"""
    # Initialize background subtractor with optimal parameters
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)
    
    # Train the background subtractor with the reference frame
    for _ in range(10):  # More training iterations for better model
        bg_subtractor.apply(reference_frame, learningRate=0.1)
    
    # Variables for motion detection and tracking
    motion_detected = False
    stability_count = 0
    best_isolated_object = None
    largest_contour_area = 0
    
    # Variables for tracking stability
    prev_center = None
    prev_contour = None
    
    print("Watching for movement... (Press ESC to exit, 'c' for manual capture)")
    
    # Prepare morphological kernels once
    open_kernel = np.ones((7, 7), np.uint8)
    close_kernel = np.ones((15, 15), np.uint8)
    
    while True:
        ret, current_frame = cap.read()
        if not ret:
            print("Error reading from camera.")
            return None, None
        
        # Apply background subtraction to get foreground mask
        fg_mask = bg_subtractor.apply(current_frame, learningRate=0)
        
        # Apply morphological operations to reduce noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, open_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_kernel)
        
        # Threshold to convert to binary mask (removing shadows)
        _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process for significant motion
        display_frame = current_frame.copy()
        largest_contour = None
        max_area = 0
        current_center = None
        
        # Find largest contour meeting minimum area requirement
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                max_area = area
                largest_contour = contour
                # Calculate center of contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    current_center = (cx, cy)
        
        # If we found a significant contour
        if largest_contour is not None:
            # Draw the largest contour
            cv2.drawContours(display_frame, [largest_contour], -1, (0, 255, 0), 2)
            
            # Create isolated object
            mask = np.zeros_like(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY))
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            # Create isolated object view (black background with colored object)
            isolated = np.zeros_like(current_frame)
            isolated[mask == 255] = current_frame[mask == 255]
            
            # Keep track of the best isolated object (largest area)
            if max_area > largest_contour_area:
                largest_contour_area = max_area
                best_isolated_object = isolated.copy()
            
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
                        cv2.destroyAllWindows()
                        return best_isolated_object, current_frame.copy()
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
        
        # Show only one view during detection as requested
        cv2.imshow("Object Detection", display_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return None, None
        elif key == ord('c'):  # Allow manual capture
            print("Manual capture triggered!")
            if best_isolated_object is not None:
                cv2.destroyAllWindows()
                return best_isolated_object, current_frame.copy()
    
    # Should never reach here due to the loop structure
    return None, None

@torch.no_grad()
def find_object_with_lightglue(isolated_object, final_frame):
    """Use LightGlue to find the object in the final frame"""
    print("Processing with LightGlue to locate the object...")
    
    # Save the images temporarily for LightGlue processing
    temp_files = ['isolated_object.jpg', 'final_frame.jpg']
    cv2.imwrite(temp_files[0], isolated_object)
    cv2.imwrite(temp_files[1], final_frame)
    
    try:
        # Use LightGlue to find the object
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize feature extractors and matcher
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        matcher = LightGlue(features='superpoint').eval().to(device)
        
        # Load images for LightGlue
        image0 = load_image(temp_files[0], device=device)
        image1 = load_image(temp_files[1], device=device)
        
        # Extract features
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)
        
        # Match features
        matches = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches = [rbd(x) for x in [feats0, feats1, matches]]
        
        # Get keypoints and matches
        kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches['matches']
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        
        # Convert tensors to numpy arrays for OpenCV
        m_kpts1_np = m_kpts1.cpu().numpy()
        
        # Draw matches on the final frame
        highlighted_frame = final_frame.copy()
        
        # Draw keypoints and matches if any were found
        if len(m_kpts1_np) > 10:  # Require minimum matches for reliability
            # Draw matched keypoints
            for pt in m_kpts1_np:
                pt_int = tuple(map(int, pt))
                cv2.circle(highlighted_frame, pt_int, 3, (0, 0, 255), -1)
            
            # Find the bounding box of matched keypoints
            x_min = max(0, int(np.min(m_kpts1_np[:, 0])))
            y_min = max(0, int(np.min(m_kpts1_np[:, 1])))
            x_max = min(highlighted_frame.shape[1], int(np.max(m_kpts1_np[:, 0])))
            y_max = min(highlighted_frame.shape[0], int(np.max(m_kpts1_np[:, 1])))
            
            # Draw bounding box around the object
            cv2.rectangle(highlighted_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return highlighted_frame
    
    except Exception as e:
        print(f"Error in LightGlue processing: {e}")
        highlighted_frame = final_frame.copy()
        return highlighted_frame
    
    finally:
        # Clean up temporary files
        for file in temp_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception:
                    pass

if __name__ == "__main__":
    main()