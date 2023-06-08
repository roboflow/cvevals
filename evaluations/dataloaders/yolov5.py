import cv2

def get_ground_truth_for_image(img_filename):
    labels = []
    masks = []
    label_file = img_filename.replace("/images/", "/labels/").replace(".jpg", ".txt")
    with open(label_file, "r") as file:
        for line in file:
            # Split the line into a list of values and convert them to floats
            values = list(map(float, line.strip().split()))
            print(values)
            if len(values) > 5:
                # normalize to bbox
                max_x = max(values[1::2])
                min_x = min(values[1::2])
                max_y = max(values[2::2])
                min_y = min(values[2::2])

                cx = (max_x + min_x) / 2
                cy = (max_y + min_y) / 2
                width = max_x - min_x
                height = max_y - min_y
                label = values[0]
            else:
                # Extract the label, and scale the coordinates and dimensions
                label = int(values[0])
                cx = values[1]
                cy = values[2]
                width = values[3]
                height = values[4]

            image = cv2.imread(img_filename)

            x0 = cx - width / 2
            y0 = cy - height / 2
            x1 = cx + width / 2
            y1 = cy + height / 2

            # scale to non-floats
            x0 = int(x0 * image.shape[1])
            y0 = int(y0 * image.shape[0])
            x1 = int(x1 * image.shape[1])
            y1 = int(y1 * image.shape[0])

            # Add the extracted data to the output list
            labels.append((x0, y0, x1, y1, label))

            if len(values) > 5:
                # Extract the mask
                mask = values[1:]

                # Add the mask to the output list
                masks.append(mask)

    return labels, masks
