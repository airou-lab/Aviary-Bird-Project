import cv2

# Load an image
image_path = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/For_Keon_3/frame_0615.jpg'  # Path to your image
image = cv2.imread(image_path)
clone = image.copy()
boxes = []

# Function to draw bounding box
def draw_box(event, x, y, flags, param):
    global boxes, image

    if event == cv2.EVENT_LBUTTONDOWN:
        boxes.append([(x, y)])

    elif event == cv2.EVENT_LBUTTONUP:
        x1, y1 = boxes[-1][0]
        x2, y2 = x, y
        # Ensure x1, y1 is the top-left and x2, y2 is the bottom-right
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        boxes[-1] = [(x1, y1), (x2, y2)]
        cv2.rectangle(image, boxes[-1][0], boxes[-1][1], (0, 255, 0), 2)
        cv2.imshow("Image", image)

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_box)

while True:
    cv2.imshow("Image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):  # Reset the image
        image = clone.copy()
        boxes = []
    elif key == ord("c"):  # Confirm the boxes
        break

cv2.destroyAllWindows()

# Save the bounding boxes to a file
with open("Feeder_and_water_3.txt", "w") as f:
    for box in boxes:
        f.write(f"{box[0][0]},{box[0][1]},{box[1][0]},{box[1][1]}\n")

print("Bounding boxes saved to boxes.txt")
