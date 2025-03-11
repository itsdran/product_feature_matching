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
    stability_threshold = 10  # Maximum movement in pixels to consider object stable
    stability_frames = 40  # Number of frames object must be stable to consider it stopped
    
    # Step 1: Capture reference frame
    reference_frame = capture_reference_frame(cap)
    if reference_frame is None:
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Step 2: Monitor for movement and capture moving object
    isolated_object, temp_file = detect_and_capture_object(
        cap, reference_frame, min_object_area, stability_threshold, stability_frames
    )
    
    if isolated_object is None:
        print("Object detection failed.")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Step 3: Initialize LightGlue and perform live tracking
    live_object_tracking(cap, isolated_object, temp_file)
    
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
def detect_and_capture_object(cap, reference_frame, min_area, stability_threshold, stability_frames_threshold):
    """Monitor live feed for movement, detect object, and wait for it to stabilize"""
    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=16, detectShadows=False)
    
    # Train the background subtractor with the reference frame
    bg_mask = bg_subtractor.apply(reference_frame)
    # Apply it again to get a better model
    for _ in range(5):
        bg_mask = bg_subtractor.apply(reference_frame, learningRate=0.1)
    
    # Variables for motion detection
    motion_detected = False
    stability_count = 0
    
    # Variables for tracking stability
    prev_center = None
    prev_contour = None
    
    print("Watching for movement... (Press ESC to exit, 'c' for manual capture)")
    
    # Prepare morphological kernels once
    kernel = np.ones((5, 5), np.uint8)
    
    while True:
        ret, current_frame = cap.read()
        if not ret:
            print("Error reading from camera.")
            return None, None
        
        # Apply background subtraction to get foreground mask
        fg_mask = bg_subtractor.apply(current_frame, learningRate=0)
        
        # Apply morphological operations to reduce noise
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
            
            if not motion_detected:
                motion_detected = True
                
            # Check if object is stable
            if prev_center is not None and current_center is not None:
                # Calculate movement distance
                distance = np.sqrt((prev_center[0] - current_center[0])**2 + 
                                  (prev_center[1] - current_center[1])**2)
                
                # Calculate contour similarity if applicable
                contour_change = 500  # Default to a large value
                if prev_contour is not None:
                    # Compare contour areas
                    area_diff = abs(cv2.contourArea(prev_contour) - cv2.contourArea(largest_contour))
                    area_percent = area_diff / max(cv2.contourArea(prev_contour), 1) * 100
                    contour_change = area_percent
                
                # Check if object is stable (minimal movement and contour change)
                if distance < stability_threshold and contour_change < 20:
                    stability_count += 1
                    
                    # Display stability progress
                    cv2.putText(display_frame, f"Stabilizing: {stability_count}/{stability_frames_threshold}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if stability_count >= stability_frames_threshold:
                        print("Object has stabilized! Capturing final stable object.")
                        
                        # Create isolated object view ONLY AFTER it has stabilized
                        isolated = np.zeros_like(current_frame)
                        mask = np.zeros_like(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY))
                        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
                        isolated[mask == 255] = current_frame[mask == 255]
                        
                        # Save the reference object from the current stable frame
                        temp_file = 'reference_object.jpg'
                        cv2.imwrite(temp_file, isolated)
                        
                        # Show the isolated object for confirmation
                        cv2.imshow("Captured Object", isolated)
                        cv2.destroyAllWindows()
                        return isolated.copy(), temp_file
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
        
        # Show only one view during detection
        cv2.imshow("Object Detection", display_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return None, None
        elif key == ord('c'):  # Allow manual capture
            print("Manual capture triggered!")
            
            # Create isolated object view for manual capture
            if largest_contour is not None:
                isolated = np.zeros_like(current_frame)
                mask = np.zeros_like(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY))
                cv2.drawContours(mask, [largest_contour], -1, 255, -1)
                isolated[mask == 255] = current_frame[mask == 255]
                
                temp_file = 'reference_object.jpg'
                cv2.imwrite(temp_file, isolated)
                
                cv2.imshow("Captured Object", isolated)
                return isolated.copy(), temp_file
    
    return None, None

@torch.no_grad()
def live_object_tracking(cap, reference_object, temp_file):
    """Perform live tracking of the object using LightGlue"""
    print("Initializing LightGlue for real-time object tracking...")
    
    try:
        # Initialize LightGlue on the appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize feature extractors and matcher
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        matcher = LightGlue(features='superpoint').eval().to(device)
        
        # Load and extract features from the reference object (only need to do this once)
        reference_image = load_image(temp_file).to(device)
        reference_feats = extractor.extract(reference_image)
        
        print("Starting live object tracking. Press 'q' to quit, 's' to save current crop.")
        
        # Frame rate control variables
        prev_time = time.time()
        fps_display_interval = 1  # update FPS every 1 second
        frame_count = 0
        fps = 0
        
        # Create a window for the tracking display
        cv2.namedWindow("Live Object Tracking", cv2.WINDOW_NORMAL)
        
        # For display purposes
        last_save_time = 0
        save_cooldown = 2  # seconds between saves to prevent multiple saves
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading from camera.")
                break
            
            # Compute FPS
            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_display_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time
            
            # Create a copy for drawing
            display_frame = frame.copy()
            
            # Convert frame to tensor for LightGlue
            # Save the current frame temporarily (this is simpler than direct conversion)
            current_frame_file = 'current_frame.jpg'
            cv2.imwrite(current_frame_file, frame)
            
            # Load the current frame with LightGlue's loader
            current_image = load_image(current_frame_file).to(device)
            
            # Extract features from current frame
            current_feats = extractor.extract(current_image)
            
            # Match features between reference object and current frame
            matches = matcher({'image0': reference_feats, 'image1': current_feats})
            
            # Process results
            processed_reference_feats, processed_current_feats, processed_matches = [
                rbd(x) for x in [reference_feats, current_feats, matches]
            ]
            
            # Get keypoints and matches
            kpts0 = processed_reference_feats['keypoints']
            kpts1 = processed_current_feats['keypoints']
            matches = processed_matches['matches']
            
            # Get matched keypoints
            if len(matches) > 0:
                m_kpts0 = kpts0[matches[..., 0]]
                m_kpts1 = kpts1[matches[..., 1]]
                
                # Convert to numpy
                m_kpts1_np = m_kpts1.cpu().numpy()
                
                # Check if we have enough matches to proceed
                if len(m_kpts1_np) >= 4:  # Need at least 4 points for a meaningful detection
                    # Find the bounding box of matched keypoints
                    x_min = max(0, int(np.min(m_kpts1_np[:, 0])))
                    y_min = max(0, int(np.min(m_kpts1_np[:, 1])))
                    x_max = min(display_frame.shape[1] - 1, int(np.max(m_kpts1_np[:, 0])))
                    y_max = min(display_frame.shape[0] - 1, int(np.max(m_kpts1_np[:, 1])))
                    
                    # Check if bounding box is valid
                    if x_max > x_min and y_max > y_min:
                        # Draw the bounding box on the display frame
                        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        
                        # Draw all matched keypoints
                        for pt in m_kpts1_np:
                            pt_int = tuple(map(int, pt))
                            cv2.circle(display_frame, pt_int, 3, (0, 0, 255), -1)
                        
                        # Add match count text
                        cv2.putText(display_frame, f"Matches: {len(m_kpts1_np)}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Handle saving when 's' is pressed
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('s') and current_time - last_save_time > save_cooldown:
                            # Extract and save the object
                            cropped_object = frame[y_min:y_max, x_min:x_max].copy()
                            save_path = f"cropped_object_{int(time.time())}.jpg"
                            cv2.imwrite(save_path, cropped_object)
                            print(f"Object saved to {save_path}")
                            last_save_time = current_time
                            
                            # Visual feedback that save occurred
                            cv2.putText(display_frame, f"Saved to {save_path}", 
                                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "Invalid bounding box", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, f"Insufficient matches: {len(m_kpts1_np)}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "No matches found", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show the tracking results
            cv2.imshow("Live Object Tracking", display_frame)
            
            # Check for exit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except Exception as e:
        print(f"Error in live tracking: {e}")
    
    finally:
        # Clean up temporary files
        for file in [temp_file, 'current_frame.jpg']:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception:
                    pass

if __name__ == "__main__":
    main()